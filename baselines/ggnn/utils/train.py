import torch
from torch.autograd import Variable
from tensorboardX import SummaryWriter

def train_ggnn(epoch, dataloader, net, criterion, optimizer, opt, writer):
    # net.train()
    
    for i, (adj_matrix, target) in enumerate(dataloader, 0):
        net.zero_grad()

        # padding = torch.zeros(len(annotation), opt.n_node, opt.state_dim - opt.annotation_dim).double()
        # init_input = torch.cat((annotation, padding), 2)
        init_input = torch.zeros(len(adj_matrix), opt.n_node, opt.state_dim).double()

        if opt.cuda:
            init_input = init_input.cuda()
            adj_matrix = adj_matrix.cuda()
            # annotation = annotation.cuda()
            target = target.cuda()

        init_input = Variable(init_input)
        adj_matrix = Variable(adj_matrix)
        # annotation = Variable(annotation)
        target = Variable(target)
        output = net(init_input, adj_matrix)
        
        #writer.add_graph(net, (init_input, adj_matrix), verbose=False)

        loss = criterion(output, target)
       
        loss.backward()
        optimizer.step()

        writer.add_scalar('loss', loss.data.item(), int(epoch))
       
        
        if i % int(len(dataloader) / 10 + 1) == 0 and opt.verbal:
            print('[%d/%d][%d/%d] Loss: %.4f' % (epoch, opt.niter, i, len(dataloader), loss.item()))

    torch.save(net, "{}/:{}".format(opt.model_path, epoch))
    #net = torch.load("{}/:{}".format(opt.model_path, epoch))

def train_biggnn(epoch, dataloader, net, criterion, optimizer, opt, writer):
    # net.train()
    
    for i, (adj_matrices, target) in enumerate(dataloader, 0):
    
        net.zero_grad()
        # optimizer.zero_grad()
        # print("------------------")
        left_adj_matrix = adj_matrices[0]
        right_adj_matrix = adj_matrices[1]
    
        left_init_input = torch.zeros(len(left_adj_matrix), opt.n_node, opt.state_dim).double()
        right_init_input = torch.zeros(len(right_adj_matrix), opt.n_node, opt.state_dim).double()

        if opt.cuda:
            # print("Using cuda for training.......")
            left_init_input = left_init_input.cuda()
            right_init_input = right_init_input.cuda()
            left_adj_matrix = left_adj_matrix.cuda()
            right_adj_matrix = right_adj_matrix.cuda()
            target = target.cuda()

        left_init_input = Variable(left_init_input)
        right_init_input = Variable(right_init_input)

        left_adj_matrix = Variable(left_adj_matrix)
        right_adj_matrix = Variable(right_adj_matrix)

        target = Variable(target)
        # print(target)

        if opt.loss == 1:
            left_output, right_output = net(left_init_input, left_adj_matrix, right_init_input, right_adj_matrix)
            #writer.add_graph(net, (left_init_input, left_adj_matrix, right_init_input, right_adj_matrix), verbose=False)
            loss = criterion(left_output,right_output, target) 
            writer.add_scalar('loss', loss.data.item(), int(epoch))
        else:
            output = net(left_init_input, left_adj_matrix, right_init_input, right_adj_matrix)
            #writer.add_graph(net, (left_init_input, left_adj_matrix, right_init_input, right_adj_matrix), verbose=False)
            loss = criterion(output, target) 
            writer.add_scalar('loss', loss.data.item(), int(epoch))
           
        loss.backward()
        optimizer.step()

        if i % int(len(dataloader) / 10 + 1) == 0 and opt.verbal:
            print('[%d/%d][%d/%d] Loss: %.4f' % (epoch, opt.niter, i, len(dataloader), loss.item()))

    print("Saving model................")
    torch.save(net, "{}/#{}".format(opt.model_path, epoch))
