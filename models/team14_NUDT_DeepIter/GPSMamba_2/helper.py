# import new Network name here and add in model_class args
from .Network import MYNET
from utils.utils import *
from tqdm import tqdm
import torch.nn.functional as F
import threading
import queue

def base_train(model, trainloader, optimizer, scheduler, epoch, args):
    tl = Averager()
    ta = Averager()
    treg = Averager()
    model = model.train()

    # 自定义加权交叉熵损失
    def _my_CE_loss(logits, targets, weights=None):
        if weights is not None:
            loss = F.cross_entropy(logits, targets, reduction='none')
            return (loss * weights).mean()
        return F.cross_entropy(logits, targets)

    for i, batch in enumerate(trainloader):
        # calculate time cost
        # timenow=time.time()
        # print('Epoch [%d/%d] Iter [%d/%d]' % (epoch + 1, args.epochs.epochs_base, i + 1, len(trainloader)))
        data, train_label = [_.cuda() for _ in batch]

        # 原始前向过程
        logits, _, _ = model(data, stochastic=args.stochastic)
        logits = logits[:, :args.num_base]
        # loss = F.cross_entropy(logits, train_label)

        # ========== 添加扰动逻辑 ==========
        perturb_output = []
        perb = 0.005  #1,3,5,7
        num_perturbations=5  # 扰动次数

        # 创建模型副本（使用深拷贝）
        # original_state = deepcopy(model.module.state_dict())
        # 创建模型副本（使用深拷贝）
        # original_model = deepcopy(model.module)
        # 保存原始模型的状态
        original_state = model.module.state_dict()


        # for _ in range(5):
        #     time_pertube=time.time()

        #     perturb_model = type(model.module)(args=args, mode=args.network.base_mode)  # 创建一个新的模型实例
        #     perturb_model.load_state_dict(original_state)  # 加载原始状态
        #     # perturb_model.load_state_dict(perturb_model.state_dict())  # 加载原始状态
        #     perturb_model = perturb_model.cuda()  # 将模型移动到 GPU

        #     # 添加参数扰动（排除bias和BN层）
        #     with torch.no_grad():
        #         for name, param in perturb_model.named_parameters():
        #             if 'bias' in name or 'bn' in name or 'fc' not in name:
        #                 continue
        #             std = perb * param.data.std()
        #             noise = torch.randn_like(param.data) * std
        #             # param.data.add_(noise)
        #             param.data = param.data + noise  # 使用非原地操作

        #     # 扰动后的前向计算
        #     perturb_logits, _, _ = perturb_model(data, stochastic=False)
        #     perturb_output.append(perturb_logits)
        #     print(f"cost time_pertube: {time.time()-time_pertube:.2f}s")
        # 计算权重

                # 创建结果队列和线程列表
        result_queue = queue.Queue()
        threads = []
        
        # 启动多个线程
        for _ in range(num_perturbations):
            thread = threading.Thread(
                target=perturb_forward,
                args=(result_queue, args,model,original_state, data, perb)
            )
            thread.start()
            threads.append(thread)
        
        # 等待所有线程完成
        for thread in threads:
            thread.join()
        
        # 收集结果
        while not result_queue.empty():
            perturb_output.append(result_queue.get().cuda())
            
        perturb_stack = torch.stack(perturb_output)
        norms = torch.norm(perturb_stack, dim=(0,2), p='nuc')
        weights = 1.0 / (norms.mean(dim=0) + 1e-8)  # 防止除以零
        weights = weights / weights.max()  # 归一化

        # 加权损失计算
        loss_proto = _my_CE_loss(logits[:, :args.num_base],
                               train_label, weights) * args.lamda_proto
        loss =  loss_proto
        # ========== 扰动逻辑结束 ==========

        acc = count_acc(logits, train_label)
        optimizer.zero_grad()
        # print(loss.grad_fn)
        loss.backward()
        optimizer.step()

        tl.add(loss.item())
        ta.add(acc)
        # print(f"cost time: {time.time()-timenow:.2f}s")


    tl = tl.item()
    ta = ta.item()
    #treg = treg.item()
    treg = 0
    return tl, ta, treg
def perturb_forward(result_queue, args,model,original_state, data, perb):
    perturb_model = type(model.module)(args=args, mode=args.network.base_mode)  # 创建一个新的模型实例
    perturb_model.load_state_dict(original_state)  # 加载原始状态
    # perturb_model.load_state_dict(perturb_model.state_dict())  # 加载原始状态
    perturb_model = perturb_model.cuda()  # 将模型移动到 GPU

    # 添加参数扰动（排除bias和BN层）
    with torch.no_grad():
        for name, param in perturb_model.named_parameters():
            if 'bias' in name or 'bn' in name or 'fc' not in name:
                continue
            std = perb * param.data.std()
            noise = torch.randn_like(param.data) * std
            # param.data.add_(noise)
            param.data = param.data + noise  # 使用非原地操作

    # 扰动后的前向计算
    perturb_logits, _, _ = perturb_model(data, stochastic=False)
    result_queue.put(perturb_logits.cpu())
def replace_base_fc(trainset, transform, model, args):
    # replace fc.weight with the embedding average of train data
    model = model.eval()

    trainloader = torch.utils.data.DataLoader(dataset=trainset, batch_size=128,
                                              num_workers=8, pin_memory=True, shuffle=False)
    trainloader.dataset.transform = transform
    embedding_list = []
    label_list = []
    # data_list=[]
    with torch.no_grad():
        for i, batch in enumerate(trainloader):
            data, label = [_.cuda() for _ in batch]
            model.module.mode = 'encoder'
            embedding, _ = model(data, stochastic = False)

            embedding_list.append(embedding.cpu())
            label_list.append(label.cpu())
    embedding_list = torch.cat(embedding_list, dim=0)
    label_list = torch.cat(label_list, dim=0)

    proto_list = []

    for class_index in range(args.base_class):
        data_index = (label_list == class_index).nonzero()
        embedding_this = embedding_list[data_index.squeeze(-1)]
        embedding_this = embedding_this.mean(0)
        proto_list.append(embedding_this)

    proto_list = torch.stack(proto_list, dim=0)

    model.module.fc.mu.data[:args.base_class] = proto_list

    return model
def replace_fc(trainset, transform, model, args, session):
    present_class = (args.base_class + session * args.way)
    previous_class = (args.base_class + (session-1) * args.way)
    # replace fc.weight with the embedding average of train data
    model = model.eval()

    trainloader = torch.utils.data.DataLoader(dataset=trainset, batch_size=128,
                                              num_workers=8, pin_memory=True, shuffle=False)
    trainloader.dataset.transform = transform
    embedding_list = []
    label_list = []
    # data_list=[]
    with torch.no_grad():
        for i, batch in enumerate(trainloader):
            data, label = [_.cuda() for _ in batch]
            model.module.mode = 'encoder'
            embedding = model(data)

            embedding_list.append(embedding.cpu())
            label_list.append(label.cpu())
    embedding_list = torch.cat(embedding_list, dim=0)
    label_list = torch.cat(label_list, dim=0)

    proto_list = []

    for class_index in range(previous_class, present_class):
        data_index = (label_list == class_index).nonzero()
        embedding_this = embedding_list[data_index.squeeze(-1)]
        embedding_this = embedding_this.mean(0)
        proto_list.append(embedding_this)

    proto_list = torch.stack(proto_list, dim=0)

    model.module.fc.mu[previous_class:present_class] = proto_list

    return model

def update_sigma_protos_feature_output(trainloader, trainset, model, args, session):
    # replace fc.weight with the embedding average of train data
    model = model.eval()
    
    trainloader = torch.utils.data.DataLoader(dataset=trainset, batch_size=128,
                                              num_workers=8, pin_memory=True, shuffle=False)
    
    
    embedding_list = []
    label_list = []
    # data_list=[]
    with torch.no_grad():
        for i, batch in enumerate(trainloader):
            data, label = [_.cuda() for _ in batch]
            #print(data.shape)
            #model.module.mode = 'encoder'
            _,embedding, _ = model(data, stochastic=False)

            embedding_list.append(embedding.cpu())
            label_list.append(label.cpu())
    embedding_list = torch.cat(embedding_list, dim=0)
    label_list = torch.cat(label_list, dim=0)

    proto_list = []
    radius = []
    if session == 0:
        
        for class_index in range(args.num_base):
            data_index = (label_list == class_index).nonzero()
            embedding_this = embedding_list[data_index.squeeze(-1)]
            #embedding_this = F.normalize(embedding_this, p=2, dim=-1)
            #print('dim of emd', embedding_this.shape)
            #print(c)
            feature_class_wise = embedding_this.numpy()
            cov = np.cov(feature_class_wise.T)
            radius.append(np.trace(cov)/64)
            embedding_this = embedding_this.mean(0)
            proto_list.append(embedding_this)
        
        args.radius = np.sqrt(np.mean(radius)) 
        args.proto_list = torch.stack(proto_list, dim=0)
    else:
        for class_index in  np.unique(trainset.targets):
            data_index = (label_list == class_index).nonzero()
            embedding_this = embedding_list[data_index.squeeze(-1)]
            #embedding_this = F.normalize(embedding_this, p=2, dim=-1)
            #print('dim of emd', embedding_this.shape)
            #print(c)
            feature_class_wise = embedding_this.numpy()
            cov = np.cov(feature_class_wise.T)
            radius.append(np.trace(cov)/64)
            embedding_this = embedding_this.mean(0)
            proto_list.append(embedding_this)
        args.proto_list = torch.cat((args.proto_list, torch.stack(proto_list, dim=0)), dim =0)

def update_sigma_novel_protos_feature_output(support_data, support_label, model, args, session):
    # replace fc.weight with the embedding average of train data
    model = model.eval()
    
    embedding_list = []
    label_list = []
    # data_list=[]
    with torch.no_grad():
        data, label = support_data, support_label
        #model.module.mode = 'encoder'
        _,embedding, _ = model(data, stochastic=False)

        embedding_list.append(embedding.cpu())
        label_list.append(label.cpu())
    embedding_list = torch.cat(embedding_list, dim=0)
    label_list = torch.cat(label_list, dim=0)

    proto_list = []
    radius = []
    assert session > 0
    for class_index in  support_label.cpu().unique():

        data_index = (label_list == class_index).nonzero()
        embedding_this = embedding_list[data_index.squeeze(-1)]
        #embedding_this = F.normalize(embedding_this, p=2, dim=-1)
        #print('dim of emd', embedding_this.shape)
        #print(c)
        feature_class_wise = embedding_this.numpy()
        cov = np.cov(feature_class_wise.T)
        radius.append(np.trace(cov)/64)
        embedding_this = embedding_this.mean(0)
        proto_list.append(embedding_this)
    args.proto_list = torch.cat((args.proto_list, torch.stack(proto_list, dim=0)), dim =0)
        
    


def test_agg(model, testloader, epoch, args, session, print_numbers=False, save_pred=False):
    test_class = args.num_base + session * args.way
    model = model.eval()
    vl = Averager()
    va = Averager()
    va_agg = Averager()
    va_agg_stochastic_agg = Averager()
    num_stoch_samples = 10

    da = DAverageMeter()
    ca = DAverageMeter()
    pred_list = []
    label_list = []
    with torch.no_grad():
        #tqdm_gen = tqdm(testloader)
        for i, batch in enumerate(testloader):
            data, test_label = [_.cuda() for _ in batch]

            
            logits, features, _ = model(data, stochastic = False)
            # logits = logits[:, :test_class]

            logits = logits[:, :test_class]


            pred = torch.argmax(logits, dim=1)
            if session == args.num_session - 1:
                pred_list.append(pred)
                label_list.append(test_label)
            loss = F.cross_entropy(logits, test_label)
            # print('logits:',logits)
            # print('test_label:',test_label)
            acc = count_acc(logits, test_label)
            # print('acc:',acc)
            vl.add(loss.item())
            va.add(acc)
            per_cls_acc, cls_sample_count = count_per_cls_acc(logits, test_label)
            da.update(per_cls_acc)
            ca.update(cls_sample_count)

            

        vl = vl.item()
        va = va.item()
        va_agg = va
        da = da.average()
        ca = ca.average()
        acc_dict = acc_utils(da, args.num_base, args.num_session, args.way, session)
    if print_numbers:
        print(acc_dict)
    return vl, va, va_agg, acc_dict

