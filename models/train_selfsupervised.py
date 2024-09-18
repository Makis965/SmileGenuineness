import torch
import numpy as np
from tqdm import tqdm
import matplotlib.pyplot as plt 
from models import Encoder, Decoder, MLP
from sklearn.metrics import accuracy_score
from datasets import FaceDataset, MaskTransform, to_float32


def img_save(img, x, set=""):
    
    img = np.clip(img[0].cpu().detach().permute(1, 2, 0).numpy() , 0, 1)
    x = np.clip(x[0].cpu().detach().permute(1, 2, 0).numpy() , 0, 1)
    
    plt.imsave(f"original{set}.png", img)
    plt.imsave(f"reconstruction{set}.png", x)
    
class SelfSupervised:
    
    
    def __init__(self, data_train_path, data_test_path, from_pretrained = None, device="cpu"):
        
        self.device = torch.device(device)
        print('device: ', self.device)

        
        self.data_train = torch.load(data_train_path)
        if data_test_path:
            self.data_test = torch.load(data_test_path)
        
        self.encoder = Encoder().to(device)
        self.decoder = Decoder().to(device)
        self.gender_mlp = MLP().to(device)
        self.smile_mlp = MLP().to(device)
        
        if from_pretrained:
            self.encoder.load_state_dict(torch.load(from_pretrained[0]))
            self.decoder.load_state_dict(torch.load(from_pretrained[1]))
            self.gender_mlp.load_state_dict(torch.load(from_pretrained[2]))
            self.smile_mlp.load_state_dict(torch.load(from_pretrained[3]))

        self.optimizer = torch.optim.Adam([
                {'params': self.encoder.parameters(), 'lr': 0.00002}, 
                {'params': self.decoder.parameters(), 'lr': 0.00002},
                {'params': self.gender_mlp.parameters(), 'lr': 0.00002},
                {'params': self.smile_mlp.parameters(), 'lr': 0.00002},
                ])
        
        self.reconstruction_loss_func = torch.nn.MSELoss()
        self.classification_loss_func = torch.nn.BCEWithLogitsLoss()

    def epoch(self):
        self.encoder.train()
        self.decoder.train()
        self.gender_mlp.train()
        self.smile_mlp.train()
        
        progress_bar = tqdm(self.data_train, desc=f"Train", unit="batch")
    
        for i, (img, l1, l2) in enumerate(self.data_train):
            
            # original image
            reference = img.to(self.device)
            
            img, l1, l2 = img.to(self.device), l1.to(self.device), l2.to(self.device)

            # moriginal image with dark rectangle in random location
            m = MaskTransform()
            input = m(img).to(self.device)

            latent_space = self.encoder(input)
            x = self.decoder(latent_space)

            reconstruction_loss = self.reconstruction_loss_func(x, reference)
            
            gender = self.gender_mlp(latent_space)
            gender_loss =self.classification_loss_func(gender.squeeze(), l1.float())
            
            smile = self.smile_mlp(latent_space)
            smile_loss = self.classification_loss_func(smile.squeeze(), l2.float())
            
            loss = 0.7 * reconstruction_loss + 0.15 * gender_loss + 0.15 * smile_loss
            
            # zero value of the gradient, prevents accumulation of gradients from different iterations
            self.optimizer.zero_grad()
            # computes the gradient of current tensor
            loss.backward() 
            # performs a single optimization step, update weights
            self.optimizer.step()
            
            # update progressbar
            progress_bar.set_postfix({"Loss": loss.item()})
            progress_bar.update()
            
            if i % 75 == 0:
                    
                img_save(img, x)
                    
                if (i % 200 == 0 )and (i != 0):
                    
                    torch.save(self.encoder.state_dict(), 'encoder.pt')
                    torch.save(self.decoder.state_dict(), 'decoder.pt')
                    torch.save(self.gender_mlp.state_dict(), 'gender_mlp.pt')
                    torch.save(self.smile_mlp.state_dict(), 'smile_mlp.pt')
        torch.save(self.encoder.state_dict(), 'encoder.pt')
        torch.save(self.decoder.state_dict(), 'decoder.pt')
        torch.save(self.gender_mlp.state_dict(), 'gender_mlp.pt')
        torch.save(self.smile_mlp.state_dict(), 'smile_mlp.pt')
                    
    def test(self):
        self.encoder.eval()
        self.decoder.eval()
        self.gender_mlp.eval()
        self.smile_mlp.eval()
        
        progress_bar = tqdm(self.data_test, desc=f"Test", unit="batch")
        genders, genders_label = [], []
        smiles, smiles_label = [], []
        reconstructions = 0
        sigm = torch.nn.Sigmoid()
    
        with torch.no_grad():
            for i, (img, l1, l2) in enumerate(self.data_test):
                img, l1, l2 = img.to(self.device), l1.numpy(), l2.numpy()
            
                genders_label.extend(l1)
                smiles_label.extend(l2)
                # original image
                reference = img.to(self.device)

                # moriginal image with dark rectangle in random location
                m = MaskTransform()
                input = m(img).to(self.device)

                latent_space = self.encoder(input)
                x = self.decoder(latent_space)
                reconstruction = self.reconstruction_loss_func(x, reference).item()

                reconstructions += reconstruction
                
                gender = torch.round(sigm(self.gender_mlp(latent_space))).cpu().detach().numpy().astype(int)
                genders.extend(gender)
                
                smile = torch.round(sigm(self.gender_mlp(latent_space))).cpu().detach().numpy().astype(int)
                smiles.extend(smile)
                
                # accuracy of current batch
                gender_a = [int(x) for x in (gender == l1)[0]]
                smile_a = [int(x) for x in (smile == l2)[0]]
                
                # update progressbar
                progress_bar.set_postfix({
                "Reconstruction": reconstruction, 
                "Gender": sum(gender_a)/len(gender_a), 
                "Smile": sum(smile_a)/len(smile_a)})
                
                progress_bar.update()
                
                if i % 100 == 0:
                        
                    img_save(img, x, "_test")
                    
        reconstruction_mse = reconstructions / len(self.data_test)
        gender_acc = accuracy_score(genders_label, genders)
        smile_acc = accuracy_score(smiles_label, smiles)
        
        print(f"Reconstrution: {reconstruction_mse}, Gencder Accuracy: {gender_acc}, Smile Accuracy: {smile_acc}")


if __name__ == "__main__":
    
    s = SelfSupervised("train.pt", "test.pt", ['encoder.pt', 'decoder.pt', 'gender_mlp.pt', 'smile_mlp.pt'], "mps")

    for epoch in range(5):
        
        print(f"Epoch: {epoch+1}")
        #s.epoch()
        s.test()