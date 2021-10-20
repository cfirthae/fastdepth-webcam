import os
import time
import csv
import numpy as np

import torch
import torch.nn.parallel
import torch.backends.cudnn as cudnn
import torch.optim
cudnn.benchmark = True

import models
from metrics import AverageMeter, Result
import utils



from torchvision import datasets, transforms
from torchvision.transforms import ToPILImage
from PIL import Image
import matplotlib.pyplot as plt
import cv2
import time

'''
from torchviz import make_dot
import onnx
import onnxruntime as ort
'''

args = utils.parse_command()
print(args)
os.environ["CUDA_VISIBLE_DEVICES"] = args.gpu # Set the GPU.

fieldnames = ['rmse', 'mae', 'delta1', 'absrel',
            'lg10', 'mse', 'delta2', 'delta3', 'data_time', 'gpu_time']
best_fieldnames = ['best_epoch'] + fieldnames
best_result = Result()
best_result.set_to_worst()

def main():
    global args, best_result, output_directory, train_csv, test_csv

    # Data loading code
    print("=> creating data loaders...")

    '''
    valdir = os.path.join('..', 'data', args.data, 'val')

    if args.data == 'nyudepthv2':
        from dataloaders.nyu import NYUDataset
        val_dataset = NYUDataset(valdir, split='val', modality=args.modality)
    else:
        raise RuntimeError('Dataset not found.')

    # set batch size to be 1 for validation
    val_loader = torch.utils.data.DataLoader(val_dataset,
        batch_size=1, shuffle=False, num_workers=args.workers, pin_memory=True)
    print("=> data loaders created.")
    '''


    # evaluation mode
    if args.evaluate:
        assert os.path.isfile(args.evaluate), \
        "=> no model found at '{}'".format(args.evaluate)
        print("=> loading model '{}'".format(args.evaluate))
        checkpoint = torch.load(args.evaluate)
        # print(checkpoint)
        if type(checkpoint) is dict:
            args.start_epoch = checkpoint['epoch']
            best_result = checkpoint['best_result']
            model = checkpoint['model']
            print("=> loaded best model (epoch {})".format(checkpoint['epoch']))
        else:
            model = checkpoint
            args.start_epoch = 0
        #print(model)

        model.eval()

        start = time.time()

        #图片输入
        filename = 'image.jpg'
        image = Image.open(filename).convert('RGB') #读取图像，转换为三维矩阵
        image = image.resize((224,224),Image.ANTIALIAS) #将其转换为要求的输入大小224*224
        transform = transforms.Compose([transforms.ToTensor()])
        img = transform(image) #转为Tensor
        x = img.resize(1,3,224,224) #如果存在要求输入图像为4维的情况，使用resize函数增加一维
        
        time1 = time.time()
        print(time1-start)

        #深度推理
        #x = torch.rand(1,3,224,224)
        x_torch = x.type(torch.cuda.FloatTensor)

        time2 = time.time()
        print(time2-start)

        depth = model(x_torch)

        time3 = time.time()
        print(time3-start)


        #图片输出
        depth_min = depth.min()
        depth_max = depth.max()
        max_val = (2**(8))-1

        if depth_max - depth_min > np.finfo("float").eps:
            out = max_val * (depth - depth_min) / (depth_max - depth_min)
        else:
            out = np.zeros(depth.shape, dtype=depth.type)
        
        out = out.cpu().detach().numpy()  # tensor转化为np.array
        out = out.reshape(224,224)  # (1,1,224,224)转为(224,224)
        #print(out)
        out = Image.fromarray(out) 
        out = out.convert('L')  # 这样才能转为灰度图，如果是彩色图则改L为‘RGB’

        end = time.time()
        print(end-start)

        out.save('depth.png')


        '''
        #转化为onnx
        input_names = ["input"]
        output_names = ["output"]
        dummy_input = torch.randn(1, 3, 224, 224, device='cuda')
        #torch.onnx.export(model, dummy_input, "FastDepth.onnx", verbose=True, input_names=input_names, output_names=output_names)


        #检验onnx转化是否正确
        print('checking onnx')
        x = x.cpu().detach().numpy()
        model = onnx.load("FastDepth.onnx")
        ort_session = ort.InferenceSession('FastDepth.onnx')
        out_onnx = ort_session.run(None, {'input': x})
        out_onnx = np.array(out_onnx).reshape(224,224)
        
        print(depth)
        print(out_onnx) 
        '''
               


        '''
        #生成结构图
        g = make_dot(y)
        g.render('fastdepth', view=False)
        print('ok')
        '''

        #output_directory = os.path.dirname(args.evaluate)
        #validate(val_loader, model, args.start_epoch, write_to_file=False)
        return



def validate(val_loader, model, epoch, write_to_file=True):
    average_meter = AverageMeter()
    model.eval() # switch to evaluate mode
    end = time.time()
    for i, (input, target) in enumerate(val_loader):
        input, target = input.cuda(), target.cuda()
        # torch.cuda.synchronize()
        data_time = time.time() - end

        # compute output
        end = time.time()
        with torch.no_grad():
            pred = model(input)
        # torch.cuda.synchronize()
        gpu_time = time.time() - end

        # measure accuracy and record loss
        result = Result()
        result.evaluate(pred.data, target.data)
        average_meter.update(result, gpu_time, data_time, input.size(0))
        end = time.time()

        # save 8 images for visualization
        skip = 50

        if args.modality == 'rgb':
            rgb = input

        if i == 0:
            img_merge = utils.merge_into_row(rgb, target, pred)
        elif (i < 8*skip) and (i % skip == 0):
            row = utils.merge_into_row(rgb, target, pred)
            img_merge = utils.add_row(img_merge, row)
        elif i == 8*skip:
            filename = output_directory + '/comparison_' + str(epoch) + '.png'
            utils.save_image(img_merge, filename)

        if (i+1) % args.print_freq == 0:
            print('Test: [{0}/{1}]\t'
                  't_GPU={gpu_time:.3f}({average.gpu_time:.3f})\n\t'
                  'RMSE={result.rmse:.2f}({average.rmse:.2f}) '
                  'MAE={result.mae:.2f}({average.mae:.2f}) '
                  'Delta1={result.delta1:.3f}({average.delta1:.3f}) '
                  'REL={result.absrel:.3f}({average.absrel:.3f}) '
                  'Lg10={result.lg10:.3f}({average.lg10:.3f}) '.format(
                   i+1, len(val_loader), gpu_time=gpu_time, result=result, average=average_meter.average()))

    avg = average_meter.average()

    print('\n*\n'
        'RMSE={average.rmse:.3f}\n'
        'MAE={average.mae:.3f}\n'
        'Delta1={average.delta1:.3f}\n'
        'REL={average.absrel:.3f}\n'
        'Lg10={average.lg10:.3f}\n'
        't_GPU={time:.3f}\n'.format(
        average=avg, time=avg.gpu_time))

    if write_to_file:
        with open(test_csv, 'a') as csvfile:
            writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
            writer.writerow({'mse': avg.mse, 'rmse': avg.rmse, 'absrel': avg.absrel, 'lg10': avg.lg10,
                'mae': avg.mae, 'delta1': avg.delta1, 'delta2': avg.delta2, 'delta3': avg.delta3,
                'data_time': avg.data_time, 'gpu_time': avg.gpu_time})
    return avg, img_merge

if __name__ == '__main__':
    main()
