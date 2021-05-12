import torch
import math
from torchvision.utils import save_image,make_grid
import os
import pytorch_ssim
from matplotlib import pyplot as plt
from sewar.full_ref import *
# from sewar.full_ref import *
import scipy.io as scio
import time
import random
from shutil import copyfile
import torchvision.transforms as tf

class Utilizer(object):
    def __init__(self,opt):
        self.opts=opt

    def getLambda(self,hdr):
        res = []
        with open(hdr) as f:
            while (1):
                line = f.readline()
                if line:
                    res.append(line)
                else:
                    break
        res = list(map(lambda i: i[0:-2], res))
        ret = []
        for item in res:
            if '.' in item:
                ret.append(float(item))
        return ret

    def plotCurve(self,losses,lres,path):
        x=list(range(len(losses)))
        plt.title('loss-curve')
        plt.xlabel('Epoch')
        plt.ylabel('loss')
        plt.plot(x,losses)
        plt.savefig(path+'loss.jpg')
        plt.clf()

        plt.title('lr-curve')
        plt.xlabel('Epoch')
        plt.ylabel('lr')
        plt.plot(x, lres)
        plt.savefig(path + 'lr.jpg')
        plt.clf()

    def psnr(self,img1,img2):
        mse = torch.mean((img1 - img2) ** 2)
        if mse < 1.0e-10:
            return 100
        return 10 * math.log10(1.0 ** 2 / mse.item())

    def ssim(self,img1,img2):
        return pytorch_ssim.ssim(img1,img2)

    def msssim(self,img1,img2):
        img1_new=self.swatchTensor(img1)
        img2_new=self.swatchTensor(img2)
        res=msssim(img1_new,img2_new,MAX=1)
        return res

    def rase(self, img1, img2):
        img1_new=self.swatchTensor(img1)
        img2_new=self.swatchTensor(img2)
        res=rase(img1_new,img2_new)
        return res

    def sam(self, img1, img2):
        img1_new=self.swatchTensor(img1)
        img2_new=self.swatchTensor(img2)
        res=sam(img1_new,img2_new)
        return res

    def scc(self,img1,img2):
        img1_new=self.swatchTensor(img1)
        img2_new=self.swatchTensor(img2)
        res=scc(img1_new,img2_new)
        return res

    def uqi(self, img1, img2):
        img1_new=self.swatchTensor(img1)
        img2_new=self.swatchTensor(img2)
        res=uqi(img1_new,img2_new)
        return res

    def vifp(self, img1, img2):
        img1_new=self.swatchTensor(img1)
        img2_new=self.swatchTensor(img2)
        res=vifp(img1_new,img2_new)
        return res

    def cosineSimilarity(self,img1,img2):
        img1=img1.squeeze()
        img2 = img2.squeeze()
        res=torch.cosine_similarity(img1,img2,dim=0)
        return torch.mean(res).item()

    def swatchTensor(self,tensor):
        B, C, H, W = tensor.size()
        new_tensor = tensor.new_tensor(tensor, device='cpu')
        array = new_tensor.numpy()
        array = np.reshape(array, (C, H, W))
        return array

    def saveimg(self,idx,low_eval,hat_eval,gt_eval,imgs_epoch):

        # low_eval=low_eval.new_tensor(low_eval,device='cpu')
        low_eval=low_eval.clone().detach().to(device='cpu')
        # hat_eval=hat_eval.new_tensor(hat_eval,device='cpu')
        hat_eval = hat_eval.clone().detach().to(device='cpu')
        gt_eval = gt_eval.clone().detach().to(device='cpu')
        # gt_eval=gt_eval.new_tensor(gt_eval,device='cpu')
        def getref(img):
            B,C,H,W=img.size()
            y=0.0
            for c in range(C):
                y+=img[:,c:c+1,:,:]**2
            y=torch.sqrt(y)
            for c in range(C):
                img[:,c:c+1,:,:]=img[:,c:c+1,:,:]/y
            return img

        def getyvalue(img):
            B, C, H, W = img.size()
            value=[]
            for c in range(C):
                value.append(torch.mean(img[:,c,:,:]))
            return value

        def compareImage(tensor_withbatch, filename):


            B, C, H, W = tensor_withbatch.size()
            for b in range(B):
                img = tensor_withbatch[b:b + 1]
                img = torch.transpose(img, 0, 1)
                save_image(img,filename,nrow=16, padding=4)

        def saveRGB(datacube,name):
            r=datacube[:,59,:,:]
            g=datacube[:,41,:,:]
            b=datacube[:,17,:,:]
            img=r
            img=torch.cat((img,g),dim=0)
            img=torch.cat((img,b),dim=0)
            r=tf.ToPILImage()(r)
            r.save(imgs_epoch+str(idx)+name+'_'+'r.jpg')
            g = tf.ToPILImage()(g)
            g.save(imgs_epoch + str(idx) + name + '_' + 'g.jpg')
            b = tf.ToPILImage()(b)
            b.save(imgs_epoch + str(idx) + name + '_' + 'b.jpg')
            # img = tf.ToPILImage()(img)
            # img.save(imgs_epoch + str(idx) + name + '_' + 'rgb.jpg')

        # compareImage(low_eval,imgs_epoch+str(idx)+'low.jpg')
        # # low_eval=torch.transpose(low_eval,0,1)
        # # save_image(low_eval,imgs_epoch+str(idx)+'low.jpg',nrow=16,padding=4)
        # compareImage(hat_eval,imgs_epoch+str(idx) + 'hat.jpg')
        # # hat_eval = torch.transpose(hat_eval, 0, 1)
        # # save_image(hat_eval, imgs_epoch+str(idx) + 'hat.jpg', nrow=16, padding=4)
        # compareImage(gt_eval,imgs_epoch+str(idx) + 'gt.jpg')
        # # gt_eval = torch.transpose(gt_eval, 0, 1)
        # # save_image(gt_eval, imgs_epoch+str(idx) + 'gt.jpg', nrow=16, padding=4)



        # low_ref=getref(low_eval)
        # hat_ref=getref(hat_eval)
        # gt_ref=getref(gt_eval)

        # low_y=getyvalue(low_ref)
        # hat_y=getyvalue(hat_ref)
        # gt_y=getyvalue(gt_ref)

        saveRGB(low_eval,'low')
        saveRGB(hat_eval, 'hat')
        saveRGB(gt_eval, 'gt')

        lambdas=self.getLambda(self.opts.hdrfile)
        plt.plot(lambdas,getyvalue(gt_eval),label='GroundTruth',color='r')
        # plt.plot(lambdas,hat_eval,label='hat',color='b')
        plt.plot(lambdas,getyvalue(low_eval),label='low',color='g')
        plt.legend()
        plt.title('SpecCurve')
        plt.xlabel('lambda')
        plt.ylabel('AveEnergy')
        plt.savefig(imgs_epoch+str(idx)+'AveEnergy.png')
        plt.clf()


        low_ref=getref(low_eval)
        # hat_ref=getref(hat_eval)
        gt_ref=getref(gt_eval)

        low_y=getyvalue(low_ref)
        # hat_y=getyvalue(hat_ref)
        gt_y=getyvalue(gt_ref)

        plt.plot(lambdas,gt_y,label='GroundTruth',color='r')
        # plt.plot(lambdas,hat_eval,label='hat',color='b')
        plt.plot(lambdas,low_y,label='low',color='g')
        plt.legend()
        plt.title('Uniform-SpecCurve')
        plt.xlabel('lambda')
        plt.ylabel('Uniform-AveEnergy')
        plt.savefig(imgs_epoch+str(idx)+'Uniform_AveEnergy.png')
        plt.clf()


    def splitdatasets(self):
        datasetGTspath=self.opts.root+self.opts.expendfile+'GT/'
        datasetlowspath = self.opts.root + self.opts.expendfile + 'low/'
        GTfilelists=os.listdir(datasetGTspath)
        lowfilelists=os.listdir(datasetlowspath)

        eval_size=int((1-self.opts.train_size)*len(GTfilelists))
        evalidxlists=set()
        while(len(evalidxlists)<=eval_size):
            evalidxlists.add(random.randint(0,len(GTfilelists)-1))

        trainGTsetpath=self.opts.root+'train/'+'GT/'
        trainlowsetpath=self.opts.root+'train/'+'low/'
        evalGTsetpath = self.opts.root + 'eval/' + 'GT/'
        evallowsetpath=self.opts.root+'eval/'+'low/'

        if not os.path.exists(trainGTsetpath):
            os.makedirs(trainGTsetpath)

        if not os.path.exists(trainlowsetpath):
            os.makedirs(trainlowsetpath)

        if not os.path.exists(evalGTsetpath):
            os.makedirs(evalGTsetpath)
        if not os.path.exists(evallowsetpath):
            os.makedirs(evallowsetpath)

        for idx in evalidxlists:
            copyfile(datasetGTspath+GTfilelists[idx],evalGTsetpath+GTfilelists[idx])
            copyfile(datasetlowspath + lowfilelists[idx], evallowsetpath + lowfilelists[idx])
            print("GeneEvalmat: "+GTfilelists[idx])

        for idx in range(len(GTfilelists)):
            if idx not in evalidxlists:
                copyfile(datasetGTspath+GTfilelists[idx],trainGTsetpath+GTfilelists[idx])
                copyfile(datasetlowspath+lowfilelists[idx],trainlowsetpath+lowfilelists[idx])
                print("GeneTrainmat: "+GTfilelists[idx])

    def geneMat(self):
        gtpath = os.path.join(self.opts.root, 'GT/')
        lowpath = os.path.join(self.opts.root, 'low/')

        if not os.path.exists(os.path.join(self.opts.root,self.opts.expendfile+'GT/')):
            os.makedirs(os.path.join(self.opts.root, self.opts.expendfile+'GT/'))

        if not os.path.exists(os.path.join(self.opts.root, self.opts.expendfile+'low/')):
            os.makedirs(os.path.join(self.opts.root, self.opts.expendfile+'low/'))

        gtnames, lownames = os.listdir(gtpath), os.listdir(lowpath)
        gtnames=sorted(list(sorted(gtnames)))
        lownames = sorted(list(sorted(lownames)))

        print("remove not mat file...")
        for n in gtnames:
            if not n.endswith('.mat'):

                os.remove(gtpath+n)
        for m in lownames:
            if not m.endswith('.mat'):
                os.remove(lowpath+m)
        print("generator 256*256 mat...")
        name = 1

        for idx in range(len(gtnames)):
            if gtnames[idx]!=lownames[idx]:
                print(str(idx)+" "+"error!")
        print("img pairs compared!")

        for idx in range(len(gtnames)):

            assert gtnames[idx] == lownames[idx]
            gt, low = scio.loadmat(gtpath + gtnames[idx])['dest'], scio.loadmat(lowpath + lownames[idx])['dest']
            start=time.time()

            new_gtpath = self.opts.root+self.opts.expendfile+'GT/' + gtnames[idx]
            new_lowpath = self.opts.root+ self.opts.expendfile+'low/' + lownames[idx]
            # low=np.resize(low,(256,256))
            # gt=np.resize(gt,(256,256))

            scio.savemat(new_lowpath, {'dest': low})
            scio.savemat(new_gtpath, {'dest': gt})

            print("Done: " + gtnames[idx]+"   "+"toc: "+str(time.time()-start))


    def psnr_ndarray(self,img1,img2):



        mse = np.mean((img1 - img2) ** 2)
        if mse < 1.0e-10:
            return 100
        return 10 * math.log10(1.0 ** 2 / mse)

    def cosineSimilartiy_ndarray(self,img1,img2):
        img1=torch.from_numpy(img1.astype(np.float32))
        img2=torch.from_numpy(img2.astype(np.float32))
        res=torch.cosine_similarity(img1,img2,dim=0)
        return torch.mean(res).item()
