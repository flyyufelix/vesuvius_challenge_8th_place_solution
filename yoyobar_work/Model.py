from .Modules import *
from .resnet3d import Resnet3d

class Model(Module):
    def __init__(self,cfg=CFG,loss_fc=nn.MSELoss()):
        super().__init__(loss_fc)
        self.cfg = cfg
        self.mean_output=cfg.mean_output
        print(cfg.backbone)
        SE=cfg.backbone.split("-")[0]=="SE"
        if cfg.backbone.split("-")[-2]=="resnet3d":
            model=Resnet3d(int(cfg.backbone.split("-")[-1]),SE)
            head=nn.Linear(model.encoder.layer4[-1].conv1.in_channels,1)
        else:
            print("error")
        self.encoder=model.encoder
        if self.mean_output:
            self.head=nn.Sequential(head,
                                nn.Sigmoid())
        else:
            self.decoder=model.decoder
        self.init_optimizer(lr=cfg.lr)

    def check_bug(self):
        self.cuda()
        x=torch.randn(self.cfg.train_batch_size,self.cfg.in_chans,self.cfg.model_input_size,self.cfg.model_input_size).cuda()
        x=self.AMP_forward(x)
        print(x.shape,x.max())

    def forward(self,x:torch.Tensor):
        #B,C,H,W=x.shape
        #x=x.permute(2,3,0,1).reshape(H*W*B,C)
        #x=normalization(x).reshape(H,W,B,C).permute(2,3,0,1)
        x=normalization(x.reshape(-1,*x.shape[2:])).reshape(x.shape)
        if x.ndim==4:
            x=x[:,None]
        if self.mean_output:
            x :tc.Tensor= self.encoder.forward(x)
            if x.ndim>2:
                x = x.mean(dim=list(range(2,x.ndim)))
            x=self.head.forward(x)
        else :
            x=self.encoder.get_each_layer_features(x)
            x=self.decoder.forward(x)
            ex=self.cfg.ex_size//2
            if ex!=0:
                x=x[...,ex:-ex,ex:-ex]
        return x
    
    def get_loss(self,y_pred:tc.Tensor,y_true:tc.Tensor):
        if self.mean_output:
            y_true = y_true.mean(dim=list(range(2,y_true.ndim)))
        return self.loss_fc(y_pred.reshape(y_true.shape),y_true)

    def predict(self,x=None,model_output=None):
        if model_output is None:
            assert x is not None
            model_output=self.forward(x)
        if self.mean_output:
            return model_output
        else :
            return tc.sigmoid(model_output)
