#Common imports
import os
import random
import copy
import numpy as np

#Pytorch
import torch
import torch.utils.data as data_utils
from torchvision import datasets, transforms

#Base Class
from .data_loader import BaseDataLoader

class MnistRotated(BaseDataLoader):
    def __init__(self, args, list_train_domains, mnist_subset, root, transform=None, data_case='train', match_func=False, download=True):
        
        super().__init__(args, list_train_domains, root, transform, data_case, match_func) 
        self.mnist_subset = mnist_subset
        self.download = download
        
        self.train_data, self.train_labels, self.train_domain, self.train_indices = self._get_data()

    def load_inds(self):
        data_dir= self.root + self.args.dataset_name + '_' + self.args.model_name + '_indices'
        if self.data_case != 'val':
            return np.load(data_dir + '/supervised_inds_' + str(self.mnist_subset) + '.npy')
        else:
            return np.load(data_dir + '/val' + '/supervised_inds_' + str(self.mnist_subset) + '.npy')
            
    def _get_data(self):
                
        if self.args.dataset_name =='rot_mnist':
            data_obj_train= datasets.MNIST(self.root,
                                        train=True,
                                        download=self.download,
                                        transform=transforms.ToTensor()
                                    )
            
            data_obj_test= datasets.MNIST(self.root,
                                        train=False,
                                        download=self.download,
                                        transform=transforms.ToTensor()
                                    )
            mnist_imgs= torch.cat((data_obj_train.data, data_obj_test.data))
            mnist_labels= torch.cat((data_obj_train.targets, data_obj_test.targets))
            
        elif self.args.dataset_name == 'fashion_mnist':
            data_obj_train= datasets.FashionMNIST(self.root,
                                                train=True,
                                                download=self.download,
                                                transform=transforms.ToTensor()
                                            )
            
            data_obj_test= datasets.FashionMNIST(self.root,
                                        train=False,
                                        download=self.download,
                                        transform=transforms.ToTensor()
                                    )
            mnist_imgs= torch.cat((data_obj_train.data, data_obj_test.data))
            mnist_labels= torch.cat((data_obj_train.targets, data_obj_test.targets))
            
        # Get total number of labeled examples
        sup_inds = self.load_inds()
        mnist_labels = mnist_labels[sup_inds]
        mnist_imgs = mnist_imgs[sup_inds]
        mnist_size = mnist_labels.shape[0] 

        to_pil=  transforms.Compose([
                transforms.ToPILImage(),
                transforms.Resize((self.args.img_w, self.args.img_h))
            ])
        
        to_augment= transforms.Compose([
                transforms.RandomResizedCrop(self.args.img_w, scale=(0.7,1.0)),
                transforms.RandomHorizontalFlip()
            ])
        
        to_tensor=  transforms.Compose([
                transforms.ToTensor(),
                transforms.Normalize((0.1307,), (0.3081,))
            ])

        # Choose subsets that should be included into the training
        training_list_img = []
        training_list_labels = []
        training_list_idx= []
        training_list_size= []

        # key: class labels, val: data indices
        indices_dict={}
        for i in range(len(mnist_imgs)):
            key= int( mnist_labels[i].numpy() )
            if key not in indices_dict.keys():
                indices_dict[key]=[]
            indices_dict[key].append( i )
        
#         if self.data_case == 'test':
#             self.list_train_domains= ['30', '45']
            
        for domain in self.list_train_domains:
            # Run transforms
            mnist_img_rot= torch.zeros((mnist_size, self.args.img_w, self.args.img_h))
            mnist_idx=[]
            
            # Shuffling the images to create random across domains
            curr_indices_dict= copy.deepcopy( indices_dict )
            for key in curr_indices_dict.keys():
                random.shuffle( curr_indices_dict[key] )
            
            for i in range(len(mnist_imgs)):
                #Rotation
                if domain == '0':
                    img_rotated= to_pil(mnist_imgs[i])
                else:
                    img_rotated= transforms.functional.rotate( to_pil(mnist_imgs[i]), int(domain) )
                    
                #Augmentation
                if self.data_case =='train' and self.args.dataset_name =="fashion_mnist":
                    mnist_img_rot[i]= to_tensor(to_augment(img_rotated))        
                else:
                    mnist_img_rot[i]= to_tensor(img_rotated)        
                    
                mnist_idx.append( i )
            
            print('Source Domain ', domain)
            training_list_img.append(mnist_img_rot)
            training_list_labels.append(mnist_labels)
            training_list_idx.append(mnist_idx)
            training_list_size.append(mnist_img_rot.shape[0])
             
        if self.match_func:
            print('Match Function Updates')
            num_classes= 10
            for y_c in range(num_classes):
                base_class_size=0
                base_class_idx=-1
                for d_idx, domain in enumerate( self.list_train_domains ):
                    class_idx= training_list_labels[d_idx] == y_c
                    curr_class_size= training_list_labels[d_idx][class_idx].shape[0]
                    if base_class_size < curr_class_size:
                        base_class_size= curr_class_size
                        base_class_idx= d_idx
                self.base_domain_size += base_class_size
                print('Max Class Size: ', base_class_size, ' Base Domain Idx: ', base_class_idx, ' Class Label: ', y_c )
                   
        # Stack
        train_imgs = torch.cat(training_list_img)
        train_labels = torch.cat(training_list_labels)
        train_indices = np.array(training_list_idx)
        train_indices= np.hstack(train_indices)
        self.training_list_size= training_list_size
        
        # Create domain labels
        train_domains = torch.zeros(train_labels.size())
        for idx in range(len(self.list_train_domains)):
            train_domains[idx * mnist_size: (idx+1) * mnist_size] += idx

        # Shuffle everything one more time
        inds = np.arange(train_labels.size()[0])
        np.random.shuffle(inds)
        train_imgs = train_imgs[inds]
        train_labels = train_labels[inds]
        train_domains = train_domains[inds].long()
        train_indices = train_indices[inds]

        # Convert to onehot
        y = torch.eye(10)
        train_labels = y[train_labels]

        # Convert to onehot
        d = torch.eye(len(self.list_train_domains))
        train_domains = d[train_domains]
        
        # If shape (B,H,W) change it to (B,C,H,W) with C=1
        if len(train_imgs.shape)==3:
            train_imgs= train_imgs.unsqueeze(1)        
        
        print('Shape: Data ', train_imgs.shape, ' Labels ', train_labels.shape, ' Domains ', train_domains.shape, ' Objects ', train_indices.shape)
        return train_imgs, train_labels, train_domains, train_indices
