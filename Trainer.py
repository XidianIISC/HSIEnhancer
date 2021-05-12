from tqdm import tqdm
import time
import torch.nn as nn
import torch.optim as optim
from torch.optim.lr_scheduler import StepLR
import numpy as np
import os
import torch

class Trainer(object):
    def __init__(self,opts,loader,model,loss,utilizer):
        self.opts=opts
        self.trainloader=loader.train_loader
        self.valloader=loader.eval_loader
        self.model=model
        self.criterion=loss
        self.utilizer=utilizer
        self.optimizer=optim.Adam(self.model.parameters(),
                                  lr=opts.initLR,
                                  betas=(0.5,0.99))
        self.sche=StepLR(self.optimizer,
                         step_size=opts.step_size,
                         gamma=opts.decray_weight)
        self.losses=[]
        self.lres=[]
        self.model=nn.DataParallel(self.model,device_ids=self.opts.device_id)
        self.model.to(self.opts.device)
        # self.model.to('cpu')
        #prepare necessary paths

        self.opts.Time='plain_unet_'+self.opts.Time

        self.ckpath=os.path.join('./',self.opts.Time,self.opts.checkpoint)
        if not os.path.exists(self.ckpath):
            os.makedirs(self.ckpath)
        self.grampath=os.path.join(self.ckpath,'gram/')
        if not os.path.exists(self.grampath):
            os.makedirs(self.grampath)
        self.imgspath=os.path.join(self.ckpath,'imgs/')
        if not os.path.exists(self.imgspath):
            os.makedirs(self.imgspath)
        self.criteriapath = os.path.join(self.ckpath, 'criteria/')
        if not os.path.exists(self.criteriapath):
            os.makedirs(self.criteriapath)
        self.saved_models = os.path.join(self.ckpath,'saved_models/')
        if not os.path.exists(self.saved_models):
            os.makedirs(self.saved_models)

    def train(self):
        print("start time: "+time.asctime(time.localtime(time.time())))
        for epoch in range(self.opts.epochs):
            loss_batch=[]
            start=time.time()

            for batch,data in enumerate(self.trainloader):
                low,gt=data['low'].to(self.opts.device),data['gt'].to(self.opts.device)
                self.optimizer.zero_grad()
                hat=self.model(low)
                loss=self.criterion(hat,gt)
                loss.backward()
                self.optimizer.step()
                loss_batch.append(loss.item())
                print("Epoch[{}/{}],Batch[{}/{}],loss:{:.8f}".format(epoch, self.opts.epochs,batch,len(self.trainloader),loss.item()))
            print("Epoch[{}/{}],loss:{:.8f}".format(epoch,self.opts.epochs,np.mean(loss_batch)))
            self.losses.append(np.mean(loss_batch))
            self.lres.append(self.optimizer.state_dict()['param_groups'][0]['lr'])
            self.sche.step(epoch)
            print("Epoch {} cost time: {:.2f}".format(epoch,time.time()-start))
            print("+"*50)
            if epoch%5==0:
                torch.save(self.model.state_dict(),self.saved_models+'EPOCH%d'%epoch+'.pth')
                self.eval(epoch)
        self.utilizer.plotCurve(self.losses,self.lres,self.grampath)
        print("End time: " + time.asctime(time.localtime(time.time())))

    def eval(self,epoch):
        #prepare necessary paths
        critera_epoch=self.criteriapath+'EPOCH%d'%epoch+'/'
        if not os.path.exists(critera_epoch):
            os.makedirs(critera_epoch)
        imgs_epoch=self.imgspath+'EPOCH%d'%epoch+'/'
        if not os.path.exists(imgs_epoch):
            os.makedirs(imgs_epoch)
        f=open(critera_epoch+'critera.txt','w')
        self.model.eval()
        # self.model.to('cpu')
        # module
        # self.model.load_state_dict({k.replace('module.',''):v for k,v in torch.load('/home/lthpc/HighSpec/2020-12-4-23-51/checkpoint/saved_models/EPOCH195.pth',map_location='cpu').items()})
        self.model.load_state_dict(torch.load('/home/lthpc/HighSpec/2020-12-4-23-51/checkpoint/saved_models/EPOCH195.pth',map_location='cpu'))
        psnr,ssim,lambdacosineSimilarity=0.0,0.0,0.0
        num_eval=len(self.valloader)
        with torch.no_grad():
            for idx,data in tqdm(enumerate(self.valloader)):
                low_eval,gt_eval=data['low'].to(self.opts.device),data['gt'].to(self.opts.device)
                # low_eval,gt_eval=data['low'],data['gt']
                hat_eval=self.model(low_eval)
                # hat_eval=hat_eval.detach()
                f.write(str(self.utilizer.psnr(gt_eval,hat_eval))+'\n')
                f.write(str(self.utilizer.ssim(gt_eval, hat_eval)) + '\n')
                f.write(str(self.utilizer.cosineSimilarity(gt_eval, hat_eval)) + '\n')

                print(str(self.utilizer.psnr(gt_eval,hat_eval))+'\n')

                psnr+=self.utilizer.psnr(gt_eval,hat_eval)
                ssim+=self.utilizer.ssim(gt_eval,hat_eval)
                lambdacosineSimilarity+=self.utilizer.cosineSimilarity(gt_eval,hat_eval)

                self.utilizer.saveimg(idx,low_eval,hat_eval,gt_eval,imgs_epoch)
            f.write('ave_psnr: '+str(psnr/num_eval)+'\n')
            f.write('ave_ssim: ' + str(ssim / num_eval)+'\n')
            f.write('ave_lambdacosineSimilarity: '+str(lambdacosineSimilarity/num_eval)+'\n')

            f.close()
        self.model.train()


    def switch2cuda(self,*args):
        return [a.to(self.opts.device) for a in args]
