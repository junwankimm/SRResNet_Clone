 ##import
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter
import argparse
from torchvision.transforms import transforms

from model import *
from dataset import *
from util import *
from matplotlib import pyplot as plt

##parser
parser = argparse.ArgumentParser(description='Train the UNet', formatter_class=argparse.ArgumentDefaultsHelpFormatter)

parser.add_argument('--lr', type=float, default=1e-4, dest='lr')
parser.add_argument('--batch_size', type=int, default=8, dest='batch_size')
parser.add_argument('--num_epoch', type=int, default=50, dest='num_epoch')
parser.add_argument('--num_workers', type=int, default=1, dest='num_workers')

parser.add_argument('--data_dir', type=str, default='../ImageRegression/datasets/BSR/BSDS500/data/images', dest='data_dir')
parser.add_argument('--ckpt_dir', type=str, default='./ckpt', dest='ckpt_dir')
parser.add_argument('--log_dir', type=str, default='./log', dest='log_dir')
parser.add_argument('--result_dir', type=str, default='./results', dest='result_dir')

parser.add_argument('--mode', default='train', type=str, dest='mode')
parser.add_argument('--train_continue', default='off', type=str, dest='train_continue')
parser.add_argument('--device', default='cuda', type=str, dest='device')

parser.add_argument('--task', default='super_resolution', choices=['denoising', 'inpainting', 'super_resolution'], type=str, dest='task')
parser.add_argument('--opts', nargs='+', default=['bilinear', 4.0, 0], dest='opts')
parser.add_argument('--ny', default=320, type=int, dest='ny')
parser.add_argument('--nx', default=480, type=int, dest='nx')
parser.add_argument('--nch', default=3, type=int, dest='nch')
parser.add_argument('--nker', default=64, type=int, dest='nker')

parser.add_argument('--network', default='srresnet', choices=['unet', 'resnet', 'autoencoder, hourglass', 'srresnet'])
parser.add_argument('--learning_type', default='plain', choices=['plain', 'residual'], type=str, dest='learning_type')

args = parser.parse_args()

learning_type = args.learning_type
network = args.network
opts = [args.opts[0], np.asarray(args.opts[1:]).astype(np.float)]
nx = args.nx
ny = args.ny
nch = args.nch
nker = args.nker
task = args.task
lr = args.lr
batch_size = args.batch_size
num_epoch = args.num_epoch
num_workers = args.num_workers

data_dir = args.data_dir
ckpt_dir = args.ckpt_dir
log_dir = args.log_dir
result_dir = args.result_dir



mode = args.mode
train_continue = args.train_continue

if args.device == 'cuda':
    if torch.cuda.is_available():
        device = 'cuda'
    else:
        raise ValueError('CUDA is not available')
elif args.device == 'mps':
    if torch.backends.mps.is_available():
        device ='mps'
    else:
        raise ValueError('MPS is not available')
else:
    device = 'cpu'

print('mode : {}'.format(mode))
print('lr : {}, batch_size : {}, num_epoch : {}, num_workers : {}, device : {}, train_continue : {}'.format(lr, batch_size, num_epoch, num_workers, device, train_continue))
##
result_dir_train = os.path.join(result_dir, 'train')
result_dir_val = os.path.join(result_dir, 'val')
result_dir_test = os.path.join(result_dir, 'test')

if not os.path.exists(result_dir):
    os.makedirs(os.path.join(result_dir_train, 'png'))
    os.makedirs(os.path.join(result_dir_val, 'png'))

    os.makedirs(os.path.join(result_dir_test, 'png'))
    os.makedirs(os.path.join(result_dir_test, 'numpy'))
##
if mode == 'train':
    transform_train = transforms.Compose([RandomCrop(shape=(ny, nx)), Normalization(), RandomFlip()])
    transform_val = transforms.Compose([RandomCrop(shape=(ny, nx)), Normalization()])

    dataset_train = Dataset(data_dir=os.path.join(data_dir, 'train'), transform=transform_train, task=task, opts=opts)
    loader_train = DataLoader(dataset_train, batch_size=batch_size, shuffle=True)

    dataset_val = Dataset(data_dir=os.path.join(data_dir, 'val'), transform=transform_val, task=task, opts=opts)
    loader_val = DataLoader(dataset_val, batch_size=batch_size, shuffle=True)

    num_data_train = len(dataset_train)
    num_data_val = len(dataset_val)

    num_batch_train = int(np.ceil(num_data_train/batch_size))
    num_batch_val = int(np.ceil(num_data_val/batch_size))
else:
    transform_test = transforms.Compose([RandomCrop(shape=(ny, nx)), Normalization()])
    dataset_test = Dataset(data_dir=os.path.join(data_dir, 'test'), transform=transform_test, task=task, opts=opts)
    loader_test = DataLoader(dataset_test, batch_size=batch_size, shuffle=False)

    num_data_test = len(dataset_test)

    num_batch_test = int(np.ceil(num_data_test / batch_size))
##
if network == 'unet':
    net = UNet(in_channels = nch, out_channels=nch, nker=nker, norm='bnorm', learning_type=learning_type).to(device)
elif network == 'hourglass':
    net = Hourglass(in_channels = nch, out_channels=nch, nker=nker, norm='bnorm', learning_type=learning_type).to(device)
elif network == 'resnet':
    net = ResNet(in_channels = nch, out_channels=nch, nker=nker, learning_type=learning_type, nblk=16).to(device)
elif network == 'srresnet':
    net = SRResNet(in_channels=nch, out_channels=nch, nker=nker, learning_type=learning_type, nblk=16).to(device)

fn_loss_BCE = nn.BCEWithLogitsLoss().to(device)
fn_loss = nn.MSELoss().to(device)
#찐 논문대로하려면
fn_loss_real = nn.BCELoss().to(device) #에 정규화텀도 추가해야함.
optim = torch.optim.Adam(net.parameters(), lr=lr)

##
fn_tonumpy = lambda x : x.to('cpu').detach().numpy().transpose(0,2,3,1)
fn_denorm = lambda x, mean, std : (x*std + mean)
fn_class = lambda x : 1.0 * (x > 0.5)
##
writer_train = SummaryWriter(os.path.join(log_dir, 'train'))
writer_val = SummaryWriter(os.path.join(log_dir, 'val'))
##
st_epoch = 0
if train_continue == 'on':
    net, optim, st_epoch = load(ckpt_dir=ckpt_dir, net=net, optim=optim)

if mode == 'train': #Training
    for epoch in range(st_epoch + 1, num_epoch + 1):
        net.train()
        loss_arr = []

        for batch, data in enumerate(loader_train, 1):

            label = data['label'].to(device)
            input = data['input'].to(device)
            output = net(input)

            optim.zero_grad()

            loss = fn_loss(output, label)
            loss.backward()

            optim.step()

            loss_arr += [loss.item()]

            print('TRAIN : EPOCH {:4d}/{:4d} | BATCH {:4d}/{:4d} | LOSS {:.4f}'.format(epoch, num_epoch, batch, num_batch_train, np.mean(loss_arr)))

            label = fn_tonumpy(fn_denorm(label, mean=0.5, std=0.5))
            input = fn_tonumpy(fn_denorm(input, mean=0.5, std=0.5))
            output = fn_tonumpy(fn_denorm(output, mean=0.5, std=0.5))

            input = np.clip(input, a_min=0, a_max=1)
            output = np.clip(output, a_min=0, a_max=1)
            id = num_batch_train*(epoch-1)+batch
            plt.imsave(os.path.join(result_dir_train, 'png', '{:04d}_label.png'.format(id)), label[0].squeeze())
            plt.imsave(os.path.join(result_dir_train, 'png', '{:04d}_input.png'.format(id)), input[0].squeeze())
            plt.imsave(os.path.join(result_dir_train, 'png', '{:04d}_output.png'.format(id)), output[0].squeeze())


            # writer_train.add_image('label', label, num_batch_train*(epoch-1) + batch, dataformats='NHWC')
            # writer_train.add_image('input', input, num_batch_train*(epoch-1) + batch, dataformats='NHWC')
            # writer_train.add_image('output', output, num_batch_train*(epoch-1) + batch, dataformats='NHWC')

        writer_train.add_scalar('loss', np.mean(loss_arr), epoch)

        ##validation

        with torch.no_grad():
            net.train()
            loss_arr = []

            for batch, data in enumerate(loader_val, 1):
                label = data['label'].to(device)
                input = data['input'].to(device)

                output = net(input)

                loss = fn_loss(output, label)
                loss_arr += [loss.item()]

                print('VALID : EPOCH {:4d}/{:4d} | BATCH {:4d}/{:4}'.format(epoch, num_epoch, batch, num_batch_val))

                label = fn_tonumpy(fn_denorm(label, mean=0.5, std=0.5))
                input = fn_tonumpy(fn_denorm(input, mean=0.5, std=0.5))
                output = fn_tonumpy(fn_denorm(output, mean=0.5, std=0.5))

                input = np.clip(input, a_min=0, a_max=1)
                output = np.clip(output, a_min=0, a_max=1)
                id = num_batch_train * (epoch - 1) + batch
                plt.imsave(os.path.join(result_dir_val, 'png', '{:04d}_label.png'.format(id)), label[0].squeeze())
                plt.imsave(os.path.join(result_dir_val, 'png', '{:04d}_input.png'.format(id)), input[0].squeeze())
                plt.imsave(os.path.join(result_dir_val, 'png', '{:04d}_output.png'.format(id)), output[0].squeeze())

                # writer_val.add_image('label', label, num_batch_val*(epoch-1) + batch, dataformats='NHWC')
                # writer_val.add_image('input', input, num_batch_val*(epoch-1) + batch, dataformats='NHWC')
                # writer_val.add_image('output', output, num_batch_val*(epoch-1) + batch, dataformats='NHWC')

        writer_val.add_scalar('loss', np.mean(loss_arr), epoch)

        if epoch % 25 == 0:
            save(ckpt_dir=ckpt_dir, net=net, optim=optim, epoch=epoch)

    writer_train.close()
    writer_val.close()
else: #Testing
    net, optim, st_epoch = load(ckpt_dir=ckpt_dir, net=net, optim=optim)
    with torch.no_grad():
        net.train()
        loss_arr = []

        for batch, data in enumerate(loader_test, 1):
            label = data['label'].to(device)
            input = data['input'].to(device)

            output = net(input)

            loss = fn_loss(output, label)
            loss_arr += [loss.item()]

            print('TEST : BATCH {:4d}/{:4} | Loss {:.4f}'.format(batch, num_batch_test, np.mean(loss_arr)))

            label = fn_tonumpy(fn_denorm(label, mean=0.5, std=0.5))
            input = fn_tonumpy(fn_denorm(input, mean=0.5, std=0.5))
            output = fn_tonumpy(fn_denorm(output, mean=0.5, std=0.5))

        for j in range(label.shape[0]):
            id = num_batch_test * (batch - 1) + j

            label_ = label[j]
            input_ = input[j]
            output_ = output[j]

            np.save(os.path.join(result_dir_test, 'numpy', '{:04d}_label.png'.format(id)), label_)
            np.save(os.path.join(result_dir_test, 'numpy', '{:04d}_input.png'.format(id)), input_)
            np.save(os.path.join(result_dir_test, 'numpy', '{:04d}_output.png'.format(id)), output_)

            input_ = np.clip(input_, a_min=0, a_max=1)
            output_ = np.clip(output_, a_min=0, a_max=1)
            label_ = np.clip(label_, a_min=0, a_max=1)

            plt.imsave(os.path.join(result_dir_test, 'png', '{:04d}_label.png'.format(id)), label_)
            plt.imsave(os.path.join(result_dir_test, 'png', '{:04d}_input.png'.format(id)), input_)
            plt.imsave(os.path.join(result_dir_test, 'png', '{:04d}_output.png'.format(id)), output_)

    print('AVERAGE TEST : BATCH {:4d}/{:4} | Loss {:.4f}'.format(batch, num_batch_test, np.mean(loss_arr)))




