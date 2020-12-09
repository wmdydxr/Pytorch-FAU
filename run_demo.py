"""
Visualize the predicted AU heatmaps and intensities for the example images (./data/F014.jpg,./data/F016.jpg, ./data/M014.jpg )
"""
import torch, numpy as np
from load_data import MyDatasets
from utils import *
from network import ResNet
from torch.utils.data import Dataset, DataLoader
import os, pickle
from torchvision.utils import save_image
import argparse
from PIL import Image

parser = argparse.ArgumentParser()
parser.add_argument('--K', default=10, help='Number of AU positions')#24,10
parser.add_argument('--dataset', default='BP4D', type=str, help='database')#BP4D,DISFA
parser.add_argument('--dataset_test', default='demo', type=str)#BP4D-val, DISFA-val
parser.add_argument('--model_path', type=str,default='./data/model.pth', help='model path')
parser.add_argument('--cuda', default='5', type=str, help='cuda')
parser.add_argument('--size', default=256, help='Image size')

def loadnet(npoints=10,path_to_model=None):
    # Load the trained model.
    net = ResNet(num_maps=npoints)
    checkpoint = torch.load(path_to_model)
    checkpoint = {k.replace('module.',''): v for k,v in checkpoint.items()}
    net.load_state_dict(checkpoint,strict=False)
    return net.to('cuda')

def predict(loader,OUT,net):
    preds = []
    with torch.no_grad():
        count = 0
        for sample in loader: 
            img = sample['Im']
            heatmap = net(img.cuda())
            out = OUT(heatmap)
            preds.append(out)  
            images = None
            maps_AU6 = None
            maps_AU10 = None
            maps_AU12= None
            maps_AU14= None
            maps_AU17= None
            threshold = 0.1
            font = cv2.FONT_HERSHEY_SIMPLEX 
            
            for (index,item) in enumerate(sample['Im'].to('cpu').detach()):
                img_ori = (255*item.permute(1,2,0).numpy()).astype(np.uint8).copy()
                AU_intensities = out[index]
                AU06_intensity = round((out[index][0]+out[index][1])/2.0,2)
                AU10_intensity = round((out[index][2]+out[index][3])/2.0,2)
                AU12_intensity = round((out[index][4]+out[index][5])/2.0,2)
                AU14_intensity = round((out[index][6]+out[index][7])/2.0,2)
                AU17_intensity = round((out[index][8]+out[index][9])/2.0,2)
                """
                Visualization of the predicted AU6 heatmap.
                """
                heatmap_AU6_0 = heatmap[index][0].to('cpu').detach() 
                heatmap_AU6_0[heatmap_AU6_0<threshold]=0.0  
                heatmap_AU6_0[heatmap_AU6_0>255*5.0]=255*5.0
                heatmap_AU6_0_np = (heatmap_AU6_0.numpy()/5.0).astype(np.uint8).copy()
                heatmap_AU6_0_rz = cv2.resize(heatmap_AU6_0_np,(256,256))
                map_AU6_0 = cv2.applyColorMap(heatmap_AU6_0_rz, cv2.COLORMAP_JET)
                map_AU6_0=cv2.cvtColor(map_AU6_0, cv2.COLOR_RGB2BGR)
                heatmap_AU6_1 = heatmap[index][1].to('cpu').detach() 
                heatmap_AU6_1[heatmap_AU6_1<threshold]=0.0  
                heatmap_AU6_1[heatmap_AU6_1>255*5.0]=255*5.0
                heatmap_AU6_1_np = (heatmap_AU6_1.numpy()/5.0).astype(np.uint8).copy()
                heatmap_AU6_1_rz = cv2.resize(heatmap_AU6_1_np,(256,256))
                map_AU6_1 = cv2.applyColorMap(heatmap_AU6_1_rz, cv2.COLORMAP_JET)
                map_AU6_1=cv2.cvtColor(map_AU6_1, cv2.COLOR_RGB2BGR)
                map_AU6 = map_AU6_0*0.5+map_AU6_1*0.5+img_ori*0.5
                cv2.putText(map_AU6,"AU6: "+str(AU06_intensity), (5, 25), font, 0.85, (255, 255, 255), 2)  
                """
                Visualization of the predicted AU10 heatmap.
                """
                heatmap_AU10_0 = heatmap[index][2].to('cpu').detach() 
                heatmap_AU10_0[heatmap_AU10_0<threshold]=0.0  
                heatmap_AU10_0[heatmap_AU10_0>255*5.0]=255*5.0
                heatmap_AU10_0_np = (heatmap_AU10_0.numpy()/5.0).astype(np.uint8).copy()
                heatmap_AU10_0_rz = cv2.resize(heatmap_AU10_0_np,(256,256))
                map_AU10_0 = cv2.applyColorMap(heatmap_AU10_0_rz, cv2.COLORMAP_JET)
                map_AU10_0=cv2.cvtColor(map_AU10_0, cv2.COLOR_RGB2BGR)
                heatmap_AU10_1 = heatmap[index][3].to('cpu').detach() 
                heatmap_AU10_1[heatmap_AU10_1<threshold]=0.0 
                heatmap_AU10_1[heatmap_AU10_1>255*5.0]=255*5.0 
                heatmap_AU10_1_np = (heatmap_AU10_1.numpy()/5.0).astype(np.uint8).copy()
                heatmap_AU10_1_rz = cv2.resize(heatmap_AU10_1_np,(256,256))
                map_AU10_1 = cv2.applyColorMap(heatmap_AU10_1_rz, cv2.COLORMAP_JET)
                map_AU10_1=cv2.cvtColor(map_AU10_1, cv2.COLOR_RGB2BGR)
                map_AU10 = map_AU10_0*0.5+map_AU10_1*0.5+img_ori*0.5
                cv2.putText(map_AU10,"AU10: "+str(AU10_intensity), (5, 25), font, 0.85, (255, 255, 255), 2)
                """
                Visualization of the predicted AU12 heatmap.
                """
                heatmap_AU12_0 = heatmap[index][4].to('cpu').detach() 
                heatmap_AU12_0[heatmap_AU12_0<threshold]=0.0  
                heatmap_AU12_0[heatmap_AU12_0>255*5.0]=255*5.0
                heatmap_AU12_0_np = (heatmap_AU12_0.numpy()/5.0).astype(np.uint8).copy()
                heatmap_AU12_0_rz = cv2.resize(heatmap_AU12_0_np,(256,256))
                map_AU12_0 = cv2.applyColorMap(heatmap_AU12_0_rz, cv2.COLORMAP_JET)
                map_AU12_0=cv2.cvtColor(map_AU12_0, cv2.COLOR_RGB2BGR)
                heatmap_AU12_1 = heatmap[index][5].to('cpu').detach() 
                heatmap_AU12_1[heatmap_AU12_1<threshold]=0.0 
                heatmap_AU12_1[heatmap_AU12_1>255*5.0]=255*5.0 
                heatmap_AU12_1_np = (heatmap_AU12_1.numpy()/5.0).astype(np.uint8).copy()
                heatmap_AU12_1_rz = cv2.resize(heatmap_AU12_1_np,(256,256))
                map_AU12_1 = cv2.applyColorMap(heatmap_AU12_1_rz, cv2.COLORMAP_JET)
                map_AU12_1=cv2.cvtColor(map_AU12_1, cv2.COLOR_RGB2BGR)
                map_AU12 = map_AU12_0*0.5+map_AU12_1*0.5+img_ori*0.5
                cv2.putText(map_AU12,"AU12: "+str(AU12_intensity), (5, 25), font, 0.85, (255, 255, 255), 2)
                """
                Visualization of the predicted AU14 heatmap.
                """
                heatmap_AU14_0 = heatmap[index][6].to('cpu').detach() 
                heatmap_AU14_0[heatmap_AU14_0<threshold]=0.0  
                heatmap_AU14_0[heatmap_AU14_0>255*5.0]=255*5.0
                heatmap_AU14_0_np = (heatmap_AU14_0.numpy()/5.0).astype(np.uint8).copy()
                heatmap_AU14_0_rz = cv2.resize(heatmap_AU14_0_np,(256,256))
                map_AU14_0 = cv2.applyColorMap(heatmap_AU14_0_rz, cv2.COLORMAP_JET)
                map_AU14_0=cv2.cvtColor(map_AU14_0, cv2.COLOR_RGB2BGR)
                heatmap_AU14_1 = heatmap[index][7].to('cpu').detach() 
                heatmap_AU14_1[heatmap_AU14_1<threshold]=0.0  
                heatmap_AU14_1[heatmap_AU14_1>255*5.0]=255*5.0
                heatmap_AU14_1_np = (heatmap_AU14_1.numpy()/5.0).astype(np.uint8).copy()
                heatmap_AU14_1_rz = cv2.resize(heatmap_AU14_1_np,(256,256))
                map_AU14_1 = cv2.applyColorMap(heatmap_AU14_1_rz, cv2.COLORMAP_JET)
                map_AU14_1=cv2.cvtColor(map_AU14_1, cv2.COLOR_RGB2BGR)
                map_AU14 = map_AU14_0*0.5+map_AU14_1*0.5+img_ori*0.5
                cv2.putText(map_AU14,"AU14: "+str(AU14_intensity), (5, 25), font, 0.85, (255, 255, 255), 2)
                """
                Visualization of the predicted AU17 heatmap.
                """
                heatmap_AU17_0 = heatmap[index][8].to('cpu').detach() 
                heatmap_AU17_0[heatmap_AU17_0<threshold]=0.0
                heatmap_AU17_0[heatmap_AU17_0>255*5.0]=255*5.0  
                heatmap_AU17_0_np = (heatmap_AU17_0.numpy()/5.0).astype(np.uint8).copy()
                heatmap_AU17_0_rz = cv2.resize(heatmap_AU17_0_np,(256,256))
                map_AU17_0 = cv2.applyColorMap(heatmap_AU17_0_rz, cv2.COLORMAP_JET)
                map_AU17_0=cv2.cvtColor(map_AU17_0, cv2.COLOR_RGB2BGR)
                heatmap_AU17_1 = heatmap[index][9].to('cpu').detach() 
                heatmap_AU17_1[heatmap_AU17_1<threshold]=0.0 
                heatmap_AU17_1[heatmap_AU17_1>255*5.0]=255*5.0 
                heatmap_AU17_1_np = (heatmap_AU17_1.numpy()/5.0).astype(np.uint8).copy()
                heatmap_AU17_1_rz = cv2.resize(heatmap_AU17_1_np,(256,256))
                map_AU17_1 = cv2.applyColorMap(heatmap_AU17_1_rz, cv2.COLORMAP_JET)
                map_AU17_1=cv2.cvtColor(map_AU17_1, cv2.COLOR_RGB2BGR)
                map_AU17 = map_AU17_0*0.5+map_AU17_1*0.5+img_ori*0.5
                cv2.putText(map_AU17,"AU17: "+str(AU17_intensity), (5, 25), font, 0.85, (255, 255, 255), 2)
                
                if images is None:
                    images = np.expand_dims(img_ori,axis=0)
                else:
                    images = np.concatenate((images, np.expand_dims(img_ori,axis=0)))

                if maps_AU6 is None:
                    maps_AU6 = np.expand_dims(map_AU6,axis=0)
                else:
                    maps_AU6 = np.concatenate((maps_AU6, np.expand_dims(map_AU6,axis=0)))
                if maps_AU10 is None:
                    maps_AU10 = np.expand_dims(map_AU10,axis=0)
                else:
                    maps_AU10 = np.concatenate((maps_AU10, np.expand_dims(map_AU10,axis=0)))
                if maps_AU12 is None:
                    maps_AU12 = np.expand_dims(map_AU12,axis=0)
                else:
                    maps_AU12 = np.concatenate((maps_AU12, np.expand_dims(map_AU12,axis=0)))
                if maps_AU14 is None:
                    maps_AU14 = np.expand_dims(map_AU14,axis=0)
                else:
                    maps_AU14 = np.concatenate((maps_AU14, np.expand_dims(map_AU14,axis=0)))
                if maps_AU17 is None:
                    maps_AU17 = np.expand_dims(map_AU17,axis=0)
                else:
                    maps_AU17 = np.concatenate((maps_AU17, np.expand_dims(map_AU17,axis=0)))
            
            # Save the visualized AU heatmaps in path "./visualize/"
            if not os.path.exists('./visualize/'):
                os.makedirs('./visualize/')
            
            save_AU6 = torch.nn.functional.interpolate(torch.from_numpy(maps_AU6/255.0).permute(0,3,1,2),scale_factor=0.5)
            save_image(save_AU6, './visualize/Subject{}_AU06.png'.format(count))
            save_AU10 = torch.nn.functional.interpolate(torch.from_numpy(maps_AU10/255.0).permute(0,3,1,2),scale_factor=0.5)
            save_image(save_AU10, './visualize/Subject{}_AU10.png'.format(count))
            save_AU12 = torch.nn.functional.interpolate(torch.from_numpy(maps_AU12/255.0).permute(0,3,1,2),scale_factor=0.5)
            save_image(save_AU12, './visualize/Subject{}_AU12.png'.format(count))
            save_AU14 = torch.nn.functional.interpolate(torch.from_numpy(maps_AU14/255.0).permute(0,3,1,2),scale_factor=0.5)
            save_image(save_AU14, './visualize/Subject{}_AU14.png'.format(count))
            save_AU17 = torch.nn.functional.interpolate(torch.from_numpy(maps_AU17/255.0).permute(0,3,1,2),scale_factor=0.5)
            save_image(save_AU17, './visualize/Subject{}_AU17.png'.format(count))            
            count += 1

    return np.concatenate(preds)

def test_epoch( dataset_test, model_path,size, npoints):
    net = loadnet(npoints,model_path)
    OUT = OutIntensity().to('cuda')
    # Load data
    database = MyDatasets(size=size,database=dataset_test)
    dbloader = DataLoader(database, batch_size=1, shuffle=False, num_workers=0, pin_memory=False)
    pred = predict(dbloader,OUT,net)
   
def main():
    global args  
    args = parser.parse_args()
    os.environ["CUDA_VISIBLE_DEVICES"] = str(args.cuda)
    test_epoch(dataset_test=args.dataset_test,model_path=args.model_path,size=args.size,npoints=args.K)

if __name__ == '__main__':
    main()