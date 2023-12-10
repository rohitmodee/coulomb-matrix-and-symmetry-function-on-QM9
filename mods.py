import numpy as np
from torch.utils.data import Dataset, DataLoader
from sklearn.metrics import mean_absolute_error, mean_squared_error

def coulombmat_eigenvalues_from_coords(atom_types, coords, padded_size):
    """
    returns sorted Coulomb matrix eigenvalues
    Args:
        atom_types : a list of atom types (single characters)
        coords : the coords as a (num_atoms x 3) numpy array
        padded_size : the number of atoms in the biggest molecule to be considered
                     anything smaller will have zeros padded to the eigenvalue list
    Returns:
        Cmat_eigenvalues : as a Numpy array
    """
    # atom_num_dict = {'C':6,'N':7,'O':8,'H':1,'F':9, 'Cl': 17, 'S': 16 }

    num_atoms = len(atom_types)

    Cmat = np.zeros((num_atoms,num_atoms))

    chargearray = np.zeros((num_atoms, 1))

    # chargearray = [atom_num_dict[str(symbol,'utf-8')] for symbol in atom_types]
    chargearray = [aa for aa in atom_types]

    for i in range(num_atoms):
        for j in range(num_atoms):
            if i == j:
                Cmat[i,j] = 0.5*chargearray[i]**2.4   # Diagonal terms
            else:
                dist = np.linalg.norm(coords[i,:] - coords[j,:])
                Cmat[i,j] = chargearray[i]*chargearray[j]/dist   #Pair-wise repulsion

    Cmat_eigenvalues = np.linalg.eigvals(Cmat)

    Cmat_eigenvalues = sorted(Cmat_eigenvalues, reverse=True) #sort (should be default)

    pad_width = padded_size - num_atoms
    Cmat_eigenvalues = np.pad(Cmat_eigenvalues, ((0, pad_width)), mode='constant')

    return Cmat_eigenvalues

class sep_ijkl_dataset(Dataset):
    def __init__(self, file):
        """
        Args:
            csv_file (string): Path to the csv file with annotations.
            root_dir (string): Directory with all the images.
        """
        self.data = np.load(file, allow_pickle=True)
        self.ener = self.data["ener"]
        batch = self.data["desc"]
        self.ener = torch.tensor([j for i in self.ener for j in i], dtype=torch.float, device=device)
        batch_size = len(self.ener)
        max_atoms = []

        for batch_idx in range(batch_size):
            lol = batch[batch_idx]
            max_atoms.append(len(lol[0])) # find longest sequence
            max_atoms.append(max([len(i) for i in lol[1]])) # find longest sequence
            max_atoms.append(max([len(i) for i in lol[2]])) # find longest sequence
            max_atoms.append(max([len(i) for i in lol[3]])) # find longest sequence
        iic = max_atoms[0::4]
        jjc = max_atoms[1::4]
        kkc = max_atoms[2::4]
        llc = max_atoms[3::4]

        des_j = []
        des_k = []
        des_l = []
        for i in range(batch_size):
            const_atom_count_i = max(iic) - iic[i]
            const_atom_count_j = max(jjc) - jjc[i]
            const_atom_count_k = max(kkc) - kkc[i]
            const_atom_count_l = max(llc) - llc[i]
            a_j = torch.zeros(const_atom_count_i, const_atom_count_j, 3)
            a_k = torch.zeros(const_atom_count_i, const_atom_count_k, 3)
            a_l = torch.zeros(const_atom_count_i, const_atom_count_l, 3)
            des_j.append(pad_sequence([torch.tensor(i) for i in batch[i][1]] + [i for i in a_j]))
            des_k.append(pad_sequence([torch.tensor(i) for i in batch[i][2]] + [i for i in a_k]))
            des_l.append(pad_sequence([torch.tensor(i) for i in batch[i][3]] + [i for i in a_l]))
        
        self.des_i = pad_sequence([torch.tensor(batch[i][0]) for i in range(batch_size)], batch_first=True).squeeze().float().to(device)
        des_j = pad_sequence(des_j, batch_first=True)
        self.des_j = torch.transpose(des_j, 1, 2).float().to(device)

        des_k = pad_sequence(des_k, batch_first=True)
        self.des_k = torch.transpose(des_k, 1, 2).float().to(device)

        des_l = pad_sequence(des_l, batch_first=True)
        self.des_l = torch.transpose(des_l, 1, 2).float().to(device)
        
    def __len__(self):
        return len(self.ener)
    
    def __getitem__(self, idx):
        sample = {"atm_i": self.des_i[idx], "atm_j": self.des_j[idx], "atm_k": self.des_k[idx], "atm_l": self.des_l[idx], "energy": self.ener[idx]}
        return sample

def all_metric(y_true, y_pred):
    print("RMSE = ", mean_squared_error(y_true, y_pred, squared=False))
    print("MAE = ", mean_absolute_error(y_true, y_pred))

    plt.figure(figsize=(6,6))
    plt.scatter(y_true, y_pred, c='crimson')
    # plt.yscale('log')
    # plt.xscale('log')

    p1 = max(max(y_pred), max(y_true))
    p2 = min(min(y_pred), min(y_true))
    plt.plot([p1, p2], [p1, p2], 'b-')
    plt.xlabel('True Values', fontsize=15)
    plt.ylabel('Predictions', fontsize=15)
    plt.axis('equal')
    plt.show()