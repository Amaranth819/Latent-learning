import numpy as np
import torch
import torch.nn as nn
import torchvision
import matplotlib.pyplot as plt
from tensorboardX import SummaryWriter

'''
    Task: add a set of numbers chosen from MNIST
'''


def create_mnist_dataset(bs, is_train):
    '''
        Return:
            images of size [bs, 1, 28, 28]
            number labels of size [bs]
    '''
    return torch.utils.data.DataLoader(
        torchvision.datasets.MNIST(
            './MNIST/', train = is_train,
            download = True, transform = torchvision.transforms.ToTensor()
        ),
        batch_size = bs,
        shuffle = True if is_train else False
    )



class DeepSetsModel(nn.Module):
    def __init__(self) -> None:
        super().__init__()

        self.net = nn.Sequential(
            nn.Flatten(),
            nn.Linear(784, 256),
            nn.Tanh(),
            nn.Linear(256, 128),
            nn.Tanh(),
            nn.Linear(128, 64)
        )

        self.add_encoder = nn.Linear(64, 1)


    def forward(self, x):
        # x: [seq_len, bs, 1, 28, 28]
        seq_len = x.size(0)
        x_i_encs = [self.net(x[i]) for i in range(seq_len)]
        x_i_encs_sum = torch.stack(x_i_encs) # [seq_len, bs, 64]
        x_i_encs_sum = torch.sum(x_i_encs_sum, 0)
        res = self.add_encoder(x_i_encs_sum).view(-1)
        return res



def train():
    device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')
    net = DeepSetsModel().to(device)
    optimizer = torch.optim.Adam(net.parameters(), lr = 1e-3)

    mini_bs = 32
    seq_len = 4
    bs = mini_bs * seq_len

    training_set = create_mnist_dataset(bs, True)
    testing_set = create_mnist_dataset(bs, False)

    root_path = './Deepsets/'
    log_path = root_path + 'log/'
    logging = SummaryWriter(log_path)

    for e in np.arange(500) + 1:
        epoch_loss_list = []

        for batch_images, batch_nums in training_set:
            batch_images = batch_images.view(seq_len, -1, 1, 28, 28).to(device)
            batch_nums = batch_nums.view(seq_len, -1).to(device)

            pred_sums = net(batch_images)
            gt_sums = batch_nums.sum(0)
            loss = torch.abs(pred_sums - gt_sums).mean()

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            epoch_loss_list.append(loss.item())

        epoch_eval_loss_list = []

        with torch.no_grad():
            for batch_images, batch_nums in testing_set:
                batch_images = batch_images.view(seq_len, -1, 1, 28, 28).to(device)
                batch_nums = batch_nums.view(seq_len, -1).to(device)

                pred_sums = net(batch_images)
                gt_sums = batch_nums.sum(0)
                loss = torch.abs(pred_sums - gt_sums).mean()
                epoch_eval_loss_list.append(loss.item())

        epoch_mean_loss = np.mean(epoch_loss_list)
        epoch_mean_eval_loss = np.mean(epoch_eval_loss_list)
        logging.add_scalar('Train/', epoch_mean_loss, e)
        logging.add_scalar('Eval/', epoch_mean_eval_loss, e)
        print('Epoch %d: Training loss = %.4f | Eval loss = %.4f' % (e, epoch_mean_loss, epoch_mean_eval_loss))

    
    state_dict = {
        'model' : net.state_dict(),
        'optimizer' : optimizer.state_dict()
    }
    torch.save(state_dict, root_path + 'model.pkl')



def vis_eval(pkl_path = './Deepsets/model.pkl', test_order_influence = False):
    device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')
    net = DeepSetsModel().to(device)
    net.load_state_dict(torch.load(pkl_path)['model'])

    mini_bs = 5
    seq_len = 4
    bs = mini_bs * seq_len
    testing_set = create_mnist_dataset(bs, True)
    with torch.no_grad():
        batch_images, batch_nums = next(iter(testing_set))
        batch_images = batch_images.view(seq_len, -1, 1, 28, 28).to(device)
        batch_nums = batch_nums.view(seq_len, -1).to(device)

        if test_order_influence:
            single_image = batch_images[:, 0, ...]
            single_num = batch_nums[:, 0]
            batch_images_tmp = []
            batch_nums_tmp = []
            for _ in range(mini_bs):
                permutation = np.random.permutation(np.arange(seq_len))
                batch_images_tmp.append(single_image[permutation])
                batch_nums_tmp.append(single_num[permutation])
            batch_images = torch.stack(batch_images_tmp, 1)
            batch_nums = torch.stack(batch_nums_tmp, 1)

        pred_sums = net(batch_images).cpu().numpy()
        gt_sums = batch_nums.sum(0).cpu().numpy()
        batch_images = batch_images.cpu().numpy()
        batch_nums = batch_nums.cpu().numpy()

    fig = plt.figure(figsize=(mini_bs * 2, seq_len * 2))
    for i in range(mini_bs * seq_len):
        fig.add_subplot(mini_bs, seq_len, i + 1)
        batch_idx = i // seq_len
        seq_idx = i % seq_len
        plt.imshow(batch_images[seq_idx][batch_idx][0])
        plt.axis('off')
        plt.title('%d' % batch_nums[seq_idx][batch_idx], y = -0.25)
    plt.show()
    plt.savefig('DeepSetsTest.png')

    print('Numbers | Predicted Sum | True Sum')
    for numbers, pred, gt in zip(np.transpose(batch_nums), pred_sums, gt_sums):
        print(numbers, pred, gt)


if __name__ == '__main__':
    vis_eval(test_order_influence = True)