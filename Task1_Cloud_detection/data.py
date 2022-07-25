# Dataset Class
class CloudDataset(Dataset):
    def __init__(self, r_dir, base_dir,train=True,pytorch=True):
        super().__init__()
        self.files = []
        for f in os.listdir(base_dir/r_dir):
            if train:
              pic_gt = cv2.imread(str(base_dir/"train_gt"/f.replace("red","gt")))
              pic_gt = cv2.cvtColor(pic_gt, cv2.COLOR_BGR2GRAY)
              to_tensor = transforms.ToTensor()
              t1 = to_tensor(pic_gt)
              ratio = t1.sum().numpy()
              if ratio/384**2 >=0.1 and ratio/384**2<=0.75:
                  self.files.append({
                      "red" : base_dir/"train_red"/f,
                      "green" : base_dir/"train_green"/f.replace("red","green"),
                      "blue" : base_dir/"train_blue"/f.replace("red","blue"),
                      "nir" : base_dir/"train_nir"/f.replace("red","nir"),
                      "gt" : base_dir/"train_gt"/f.replace("red","gt"),
                  })
            else:
              self.files.append({
                  "red" : base_dir/"train_red"/f,
                  "green" : base_dir/"train_green"/f.replace("red","green"),
                  "blue" : base_dir/"train_blue"/f.replace("red","blue"),
                  "nir" : base_dir/"train_nir"/f.replace("red","nir"),
                  "gt" : base_dir/"train_gt"/f.replace("red","gt"),
              })
        
    def split_train_test(self, ratio = 0.7):
        self.shuffle()
        elements = len(self.files)
        files = self.files[:math.ceil(elements * 0.7)]
        middle = int(len(files) * ratio)
        self.train_files = files[:middle]
        test_files = files[middle:]
        middle = middle = int(len(test_files) * 0.5)
        self.val_files = test_files[:middle]
        self.test_files = test_files[middle:]
        print(f"{len(self.train_files)} ---- {len(self.test_files)} ---- {len(self.val_files)}")
        return self.train_files, self.val_files,self.test_files
                                       
    def __len__(self):
        return len(self.files)

    def __getitem__(self,i):
      pic_red = cv2.imread(str(self.files[i]["red"]))
      pic_red = cv2.cvtColor(pic_red, cv2.COLOR_BGR2GRAY)
      pic_green = cv2.imread(str(self.files[i]["green"]))
      pic_green = cv2.cvtColor(pic_green, cv2.COLOR_BGR2GRAY)
      pic_blue = cv2.imread(str(self.files[i]["blue"]))
      pic_blue = cv2.cvtColor(pic_blue, cv2.COLOR_BGR2GRAY)
      print("here : ", str(self.files[i]["gt"]))
      pic_gt = cv2.imread(str(self.files[i]["gt"]))
      pic_gt = cv2.cvtColor(pic_gt, cv2.COLOR_BGR2GRAY)
      pic_nir = cv2.imread(str(self.files[i]["nir"]))
      pic_nir = cv2.cvtColor(pic_nir, cv2.COLOR_BGR2GRAY)
      pic = (np.dstack((pic_green,pic_blue,pic_red))) .astype(np.uint8)
      return pic,pic_gt

    def shuffle(self):
        return random.shuffle(self.files)

class _Data(Dataset):
  def __init__(self, files):
      self.files = files
      self.images = []
      for f in files:
          pic_red = cv2.imread(str(f["red"]))[...,0]
          pic_green = cv2.imread(str(f["green"]))[...,0]
          pic_blue = cv2.imread(str(f["blue"]))[...,0]
          pic_nir = cv2.imread(str(f["nir"]))[...,0]
          pic_gt = cv2.imread(str(f["gt"]))[...,0]
          raw_rgb = np.stack(
              [pic_red, pic_green, pic_blue, pic_nir]
              , axis=2)
          img = Image.fromarray((raw_rgb).astype(np.uint8))
          self.images.append((torch.from_numpy(np.asarray(img).transpose((2,0,1))),torch.from_numpy(pic_gt)))
      print("Done ✅")
  def __getitem__(self,i):
      return self.images[i]

  def __len__(self):
      return len(self.images)   