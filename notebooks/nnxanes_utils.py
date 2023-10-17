# Standard libraries
import os
import re

# Libraries related to data manupulation
import pandas as pd
import numpy as np
from glob import glob

# Libraries related to plotting
import matplotlib.pyplot as plt
import scienceplots
import matplotlib.ticker as ticker
import py3Dmol


# Libraries related to coordination number calculation

from scipy.spatial import distance

plt.style.use(["science", "nature", "bright"])
font_size = 8
plt.rcParams.update({"font.size": font_size})
plt.rcParams.update({"axes.labelsize": font_size})
plt.rcParams.update({"xtick.labelsize": font_size})
plt.rcParams.update({"ytick.labelsize": font_size})
plt.rcParams.update({"legend.fontsize": font_size})

# Directory that contains the data

dir_path = "IV_Machine_learning_demo/NN-XANES-data/nanoparticles/"
data_dir = "IV_Machine_learning_demo/NN-XANES-data/"
sample_dir = ""
feff_file = "feff_for_Pt_bulk.dat"
fdmnes_file = "fdmnes_for_Pt_bulk_conv.txt"
experimnet_file = "exp_xanes.nor"

# CONSTANTS: FDMNES energy shift and scale factors

fdmnes_scale_factors = 0.036
fdmnes_energy_shift = 11563

# CONSTANTS: Range of the nearest neighbor distances
NN_RANGE = [
    [0, 3.5],
    [3.5, 4.4],
    [4.4, 5.2],
    [5.2, 6.0],
]

# CONSTANTS: Maximum coordination numbers of the bulk materials
bulk_CNs = np.array([12, 6, 24, 12])

# FEFF keywords used for parsing

feff_keywords = [
    "ATOMS",
    "POTENTIALS",
    "RECIPROCAL",
    "REAL",
    "CIF",
    "LATTICE",
    "TARGET",
    "TITLE",
    "COORDINATES",
    "RMULTIPLIER",
    "SGROUP",
    "CFAVERAGE",
    "OVERLAP",
    "EQUIVALENCE",
    "EXAFS",
    "ELNES",
    "EXELFS",
    "LDOS",
    "XANES",
    "ELLIPTICITY",
    "MULTIPOLE",
    "POLARIZATION",
    "COMPTON",
    "DANES",
    "FPRIME",
    "MDFF",
    "NRIXS",
    "XES",
    "XNCD",
    "CONTROL",
    "END",
    "KMESH",
    "PRINT",
    "DIMS",
    "EGRID",
    "AFOLP",
    "COREHOLE",
    "EDGE",
    "SCF",
    "S02",
    "CHBROAD",
    "CONFIG",
    "EXCHANGE",
    "FOLP",
    "HOLE",
    "NOHOLE",
    "RGRID",
    "UNFREEZEF",
    "CHSHIFT",
    "CHWIDTH",
    "CORVAL",
    "EGAP",
    "EPS0",
    "EXTPOT",
    "INTERSTITIAL",
    "ION",
    "JUMPRM",
    "NUMDENS",
    "OPCONS",
    "PREP",
    "RESTART",
    "SCREEN",
    "SETE",
    "SPIN",
    "LJMAX",
    "LDEC",
    "MPSE",
    "PLASMON",
    "PMBSE",
    "RPHASES",
    "RSIGMA",
    "TDLDA",
    "FMS",
    "DEBYE",
    "BANDSTRUCTURE",
    "STRFACTORS",
    "RPATH",
    "NLEG",
    "PCRITERIA",
    "SS",
    "SYMMETRY",
    "CRITERIA",
    "IORDER",
    "NSTAR",
    "ABSOLUTE",
    "CORRECTIONS",
    "SIG2",
    "SIG3",
    "SIGGK",
    "MBCONV",
    "SFCONV",
    "RCONV",
    "SELF",
    "SFSE",
    "CGRID",
    "RHOZZP",
    "MAGIC",
]


# Functions to read FEFF and FDMNES files


def read_xyz_from_feff(feff_input: str) -> tuple[int, np.ndarray]:
    """Reads the coordinates from a FEFF input string

    Args:
        feff_input (str): FEFF input string

    Returns:
        absorb_index (int): Index of the absorbing atom
        coordinates (np.ndarray): Coordinates of the atoms
    """

    feff_keywords_pattern = "|".join(feff_keywords)
    comment_pattern = re.compile(r"\*.*\n")
    ATOMS_pattern = re.compile(
        r"ATOMS\n(.*)\n(?:{})".format(feff_keywords_pattern), re.DOTALL
    )

    text_without_comments = re.sub(comment_pattern, "", feff_input)
    atoms_text = re.findall(ATOMS_pattern, text_without_comments)[0]
    atoms_text = atoms_text.split("\n")

    coordinates = []
    absorb_index = 0

    for line in atoms_text:
        position = line.split()

        if len(position) > 3:
            coordinates.append(position[:3])

            if position[3] == "0":
                absorb_index = len(coordinates) - 1

    return absorb_index, np.array(coordinates, dtype=float)


def read_xyz_from_feff_file(filename: str) -> tuple[int, np.ndarray]:
    """Reads the coordinates from a FEFF input file

    Args:
        filename (str): FEFF input file path

    Returns:
        absorb_index (int): Index of the absorbing atom
        coordinates (np.ndarray): Coordinates of the atoms
    """

    if not os.path.isfile(filename):
        raise FileNotFoundError("File {} not found".format(filename))

    with open(filename, "r") as f:
        feff_input = f.read()

    return read_xyz_from_feff(feff_input)


def get_XAS_fdmnes(filename: str, w: int = 1) -> np.ndarray:
    """Reads the XAS data from a FDMNES output file

    Args:
        filename (str): FDMNES output file path
        w (int, optional): Weight of the XAS spectrum. Defaults to 1.

    Returns:
        XAS_spectrum(np.ndarray): XAS data. First column is the energy, second column is the normalized absorption coefficients.
    """

    fdmnes_data = np.loadtxt(filename, skiprows=1)
    fdmnes_data[:, 0] += fdmnes_energy_shift
    fdmnes_data[:, 1] /= fdmnes_scale_factors * w
    return fdmnes_data


def get_XAS_feff(filename: str) -> np.ndarray:
    """Reads the XAS data from a FEFF output file

    Args:
        filename (str): FEFF output file path

    Returns:
        XAS_spectrum(np.ndarray): XAS data. First column is the energy, second column is the normalized absorption coefficients.
    """
    feff_data = np.loadtxt(filename, skiprows=1)
    return feff_data[:, [0, 3]]


def get_XAS_experiment(filename: str, i: int, dele: float = 0.0) -> np.ndarray:
    """Reads the XAS data from an experimental file

    Args:
        filename (str): Experimental file path
        i (int): Column index of the absorption coefficients
        dele (float, optional): Energy shift. Defaults to 0.0. The energy will be calculated as energy = energy - dele.

    Returns:
        XAS_spectrum(np.ndarray): XAS data. First column is the energy, second column is the normalized absorption coefficients.
    """
    experiment_data = np.loadtxt(filename, skiprows=1)
    experiment_data[:, 0] -= dele
    return experiment_data[:, [0, i]]


def w(filename, n):
    """Obtain weight from the fdmnes output file

    Args:
        filename (str): FDMNES output file path
        n (int): Index of site

    Returns:
        weight (float): Multiplicity of the site
    """

    # n=0
    with open(filename, "r") as f:
        text = f.readlines()

    for line in text:
        if line.startswith("ipr"):
            if n == 0:
                return float(line.split()[-1])

            n -= 1


# Functions to calculate Coordination numbers


# some efficient ways to calculate distance matrix
def dist_matrix(xyz: np.ndarray, absorb_index: int = None) -> np.ndarray:
    """Calculate the distance matrix of the atoms

    Args:
        xyz (np.ndarray): Coordinates of the atoms
        absorb_index (int, optional): Index of the absorbing atom. Defaults to None.

    Returns:
        np.ndarray: Distance matrix of xyz coordinates. If absorb_index is None, the shape of the matrix is (N, N), where N is the number of atoms. If absorb_index is not None, the shape of the matrix is (1, N).
    """
    if absorb_index is None:
        return distance.cdist(xyz, xyz, "euclidean")
        # d = xyz[:,None,:]-xyz
        # return np.sqrt(np.einsum('ijk,ijk->ij',d,d))
    else:
        return distance.cdist(xyz[[absorb_index]], xyz, "euclidean")
        # d = xyz[absorb_index] - xyz
        # return np.sqrt(np.einsum('ij,ij->i',d,d))


def get_NN_from_xyz(
    xyz: np.ndarray, absorb_index: int = None, NN_range: list[list] = None
):
    """Calculate the nearest neighbor distribution of the atoms

    Args:
        xyz (np.ndarray): Coordinates of the atoms
        absorb_index (int, optional): Index of the absorbing atom. Defaults to None.
        NN_range (list[list], optional): Range of the nearest neighbor distribution. Defaults to None.

    Returns:
        np.ndarray: Nearest neighbor distribution of the atoms. If absorb_index in None, particle average NN is returned. If absorb_index is not None, site specific NN is returned.
    """
    if NN_range is None:
        NN_range = NN_RANGE

    dist_mat = dist_matrix(xyz, absorb_index)

    if absorb_index is not None:
        atom_num = 1
    else:
        atom_num = len(xyz)

    NN = []

    for i in range(len(NN_range)):
        NN_tmp = dist_mat[(dist_mat > NN_range[i][0]) & (dist_mat <= NN_range[i][1])]
        NN.append(len(NN_tmp) / atom_num)

    return np.array(NN)


# Functions to read data from directories


def obtain_spectra_and_CN_single_site(dir_path: str, n: int = 0):
    """Obtain XAS spectra, coordination number and coordinates from a directory

    Args:
        dir_path (str): Directory path
        n (int, optional): Index of site. Defaults to 0.

    Returns:
        [fdmnes_data, feff_data, CN, file_xyz]: [FDMNES XAS data, FEFF XAS data, coordination number, file path to coordinates]
    """
    file_xyz = glob(os.path.join(dir_path, "*.xyz"))[0]
    feff_inp = os.path.join(dir_path, "feff.inp")
    feff_out = os.path.join(dir_path, "xmu.dat")
    fdmnes_file = os.path.join(dir_path, "out_fdmnes_conv.txt")
    fdmnes_bav_file = os.path.join(dir_path, "out_fdmnes_bav.txt")
    absorb_index, positions = read_xyz_from_feff_file(feff_inp)
    weight = w(fdmnes_bav_file, n)
    fdmnes_data = get_XAS_fdmnes(fdmnes_file, weight)
    feff_data = get_XAS_feff(feff_out)
    CN = get_NN_from_xyz(positions, absorb_index)
    return [fdmnes_data, feff_data, CN, file_xyz]


def obtain_spectra_and_CN(dir_path):
    """Obtain XAS spectra, coordination number and coordinates from a directory

    Args:
        dir_path (str): Directory path

    Returns:
        spectra_and_CN (list): List of [FDMNES XAS data, FEFF XAS data, coordination number, file path to coordinates]
    """
    particle_name_pattern = re.compile(r".*/(.+)/(.*)/")

    dir_list = glob(os.path.join(dir_path, "*/*/"))

    spectra_and_CN = []

    for directory in dir_list:
        particle_regex = re.findall(particle_name_pattern, directory)[0]
        particle = particle_regex[0]
        site = particle_regex[1]

        directory = os.path.join(dir_path, particle, site)

        spectra_and_CN.append(
            obtain_spectra_and_CN_single_site(directory, int(site) - 1)
        )

    return spectra_and_CN


# Functinos to plot XAS spectra


def plot_XANES(spectra, labels, E0=11563, E_range=None):
    if E_range is None:
        E_range = [E0 - 20, E0 + 80]

    fig, ax = plt.subplots(figsize=(3, 3))

    # set enegy range for E
    E0 = 11563
    E_range = [E0 - 20, E0 + 80]

    for i in range(len(spectra)):
        ax.plot(spectra[i][:, 0], spectra[i][:, 1], label=labels[i])

    ax.set_xlabel("Energy (eV)")
    ax.set_ylabel("Normalized absorption")

    ax.set_xlim(E_range)

    ax.legend()

    fig.tight_layout(pad=0.5)

    fig.show()


def plot_spectra_and_CN(
    spectra_and_CN, index_list, E0=11563, E_range=None, foilexperiment=None
):
    E0 = 11563
    if E_range is None:
        E_range = [E0 - 20, E0 + 80]

    fig, ax = plt.subplots(len(index_list), 2, figsize=(6, 3 * (len(index_list))))
    ax = ax.flatten()

    for i in range(len(index_list)):
        fig.text(0, 1 - i / len(index_list), f"Spectrum {index_list[i]}")

        ax[i * 2 + 0].plot(
            spectra_and_CN[i][0][:, 0], spectra_and_CN[i][0][:, 1], label="FDMNES"
        )
        ax[i * 2 + 1].plot(
            spectra_and_CN[i][1][:, 0], spectra_and_CN[i][1][:, 1], label="FEFF"
        )

        if foilexperiment is not None:
            ax[i * 2 + 0].plot(
                foilexperiment[:, 0], foilexperiment[:, 1], label="Pt Foil (Exp)"
            )
            ax[i * 2 + 1].plot(
                foilexperiment[:, 0], foilexperiment[:, 1], label="Pt Foil (Exp)"
            )

        y_max = max(spectra_and_CN[i][0][:, 1].max(), spectra_and_CN[i][1][:, 1].max())

        for j in range(2):
            ax[i * 2 + j].set_xlabel("Energy (eV)")
            ax[i * 2 + j].set_ylabel("Normalized absorption")
            ax[i * 2 + j].set_ylim(0, y_max * 1.1)
            ax[i * 2 + j].set_xlim(E_range)
            ax[i * 2 + j].legend()

    fig.tight_layout(h_pad=1.5, w_pad=0.5)
    fig.show()


def plot_molecule(file_xyz):
    with open(file_xyz, "r") as f:
        xyz = f.read()

    view = py3Dmol.view(width=400, height=400)
    view.addModel(xyz, "xyz")
    view.setStyle({"sphere": {"scale": 0.4}, "stick": {"scale": 0.2}})
    # view.setStyle({'sphere':{'scale': 0.4}})
    view.zoomTo()
    return view


def plot_spectra_and_CN_model(spectra_and_CN, index_list):
    for index in index_list:
        print("Spectrum ID:", index)
        print("Coordination numbers:", spectra_and_CN[index][2])
        print("File path:", spectra_and_CN[index][3])
        plot_molecule(spectra_and_CN[index][3]).show()


def plot_spectra_and_CN_stack(spectra_and_CN, index_list, E0=11563, E_range=None):
    E0 = 11563
    if E_range is None:
        E_range = [E0 - 20, E0 + 80]

    fig, ax = plt.subplots(len(index_list), 1, figsize=(3, 3 * (len(index_list))))
    ax = ax.flatten()

    for i in range(len(index_list)):
        fig.text(0, 1 - i / len(index_list), f"Spectrum {index_list[i]}")

        ax[i].plot(
            spectra_and_CN[i][0][:, 0], spectra_and_CN[i][0][:, 1], label="FDMNES"
        )
        ax[i].plot(spectra_and_CN[i][1][:, 0], spectra_and_CN[i][1][:, 1], label="FEFF")

        y_max = max(spectra_and_CN[i][0][:, 1].max(), spectra_and_CN[i][1][:, 1].max())

        ax[i].set_xlabel("Energy (eV)")
        ax[i].set_ylabel("Normalized absorption")
        ax[i].set_ylim(0, y_max * 1.1)
        ax[i].set_xlim(E_range)
        ax[i].legend()

    fig.tight_layout(h_pad=1.5, w_pad=0.5)
    fig.show()


def read_and_prepare_spectra_list():
    """Function to read and prepare the spectra list

    This function is specific to the data in the directory "IV_Machine_learning_demo/NN-XANES-data/nanoparticles/"
    It needs to be modified if the data is stored in different format or directory.

    Returns:
        dict:
            spectra_list: List of [FDMNES XAS data, FEFF XAS data, coordination number, file path to coordinates]
            bulk_CNs: Maximum coordination numbers of the bulk materials
            foilFEFFi: Interpolated FEFF XAS data
            foilFDMNESi: Interpolated FDMNES XAS data
            foilexperimenti: Interpolated experimental XAS data
            emesh: Energy mesh
    """
    spectra_and_CN = obtain_spectra_and_CN(dir_path)

    foilFEFF = get_XAS_feff(os.path.join(data_dir, sample_dir, feff_file))
    foilFDMNES = get_XAS_fdmnes(os.path.join(data_dir, sample_dir, fdmnes_file), 4)
    foilexperiment = get_XAS_experiment(
        os.path.join(data_dir, sample_dir, experimnet_file), 1, 3
    )

    min_feff = np.min([spectra_and_CN[i][1][0][0] for i in range(len(spectra_and_CN))])
    max_feff = np.min([spectra_and_CN[i][1][-1][0] for i in range(len(spectra_and_CN))])
    min_fdmnes = np.max(
        [spectra_and_CN[i][0][0][0] for i in range(len(spectra_and_CN))]
    )
    max_fdmnes = np.max(
        [spectra_and_CN[i][0][-1][0] for i in range(len(spectra_and_CN))]
    )

    print(f"Energy range of FEFF: {min_feff} - {max_feff}")
    print(f"Energy range of FDMNES: {min_fdmnes} - {max_fdmnes}")

    # caculate the energy range of the spectra, which contains all the spectra
    emin = np.max([min_feff, min_fdmnes, foilFDMNES[0][0], foilFEFF[0][0]])
    emax = np.min([max_feff, max_fdmnes, foilFDMNES[-1][0], foilFEFF[-1][0]])
    emesharray = spectra_and_CN[0][0][:, 0]
    emesh = emesharray[np.logical_and(emesharray <= emax, emesharray >= emin)]

    # interpolate the spectra to the same energy mesh (emesh)
    foilFEFFi = np.interp(emesh, foilFEFF[:, 0], foilFEFF[:, 1])
    foilFDMNESi = np.interp(emesh, foilFDMNES[:, 0], foilFDMNES[:, 1])
    foilexperimenti = np.interp(emesh, foilexperiment[:, 0], foilexperiment[:, 1])

    # interpolate all of the spectra to same energy mesh
    for i in range(len(spectra_and_CN)):
        spectra_and_CN[i][0] = np.stack(
            [
                emesh,
                np.interp(
                    emesh, spectra_and_CN[i][0][:, 0], spectra_and_CN[i][0][:, 1]
                ),
            ]
        ).T
        spectra_and_CN[i][1] = np.stack(
            [
                emesh,
                np.interp(
                    emesh, spectra_and_CN[i][1][:, 0], spectra_and_CN[i][1][:, 1]
                ),
            ]
        ).T

    return {
        "spectra_list": spectra_and_CN,
        "bulk_CNs": bulk_CNs,
        "foilFEFFi": foilFEFFi,
        "foilFDMNESi": foilFDMNESi,
        "foilexperimenti": foilexperimenti,
        "emesh": emesh,
    }


def linear_combination_FDMNES(n: int, spectra: list):
    """Calculate the averaged spectrum and averaged coordination number of n random FDMNES spectra

    Args:
        n (int): Number of spectra to calculate averaged spectrum
        spectra (list): List of [FDMNES XAS data, FEFF XAS data, coordination number, file path to coordinates]

    Returns:
        spectra_ave (np.ndarray): Averaged spectrum
        coordinates_ave (np.ndarray): Averaged coordination number
    """
    num = np.random.randint(0, len(spectra), n)
    spectra_tmp = []
    coordinates_tmp = []

    for i in num:
        spectra_tmp.append(spectra[i][0][:, 1])
        coordinates_tmp.append(spectra[i][2])

    spectra_ave = np.mean(spectra_tmp, axis=0) - foilFDMNESi
    coordinates_ave = np.mean(coordinates_tmp, axis=0) / bulk_CNs

    return spectra_ave, coordinates_ave


def linear_combination_FEFF(n: int, spectra: list):
    """Calculate the averaged spectrum and averaged coordination number of n random FEFF spectra

    Args:
        n (int): Number of spectra to calculate averaged spectrum
        spe ctrum (list): List of [FDMNES XAS data, FEFF XAS data, coordination number, file path to coordinates]

    Returns:
        spectra_ave (np.ndarray): Averaged spectrum
        coordinates_ave (np.ndarray): Averaged coordination number
    """
    num = np.random.randint(0, len(spectra), n)
    spectra_tmp = []
    coordinates_tmp = []

    for i in num:
        spectra_tmp.append(spectra[i][1][:, 1])
        coordinates_tmp.append(spectra[i][2])

    spectra_ave = np.mean(spectra_tmp, axis=0) - foilFEFFi
    coordinates_ave = np.mean(coordinates_tmp, axis=0) / bulk_CNs

    return spectra_ave, coordinates_ave


def gen_training_FDMNES(num: int, n: int, spectra: list):
    """Generate training data for FDMNES spectra

    Args:
        num (int): Number of training data to be generated
        n (int): Number of spectra to calculate averaged spectrum
        spectra (list): List of [FDMNES XAS data, FEFF XAS data, coordination number, file path to coordinates]

    Returns:
        data_x (list): List of averaged spectra
        data_y (list): List of averaged coordination numbers
    """
    data_x = []
    data_y = []

    for i in range(num):
        data = linear_combination_FDMNES(n, spectra)
        data_x.append(data[0])
        data_y.append(data[1])
    return data_x, data_y


def gen_training_FEFF(num, n, spectrums):
    """Generate training data for FEFF spectra

    Args:
        num (int): Number of training data to be generated
        n (int): Number of spectra to calculate averaged spectrum
        spectra (list): List of [FDMNES XAS data, FEFF XAS data, coordination number, file path to coordinates]

    Returns:
        data_x (list): List of averaged spectra
        data_y (list): List of averaged coordination numbers
    """
    data_x = []
    data_y = []

    for i in range(num):
        data = linear_combination_FEFF(n, spectrums)
        data_x.append(data[0])
        data_y.append(data[1])
    return data_x, data_y


element_to_Z = {
    "H": 1,
    "He": 2,
    "Li": 3,
    "Be": 4,
    "B": 5,
    "C": 6,
    "N": 7,
    "O": 8,
    "F": 9,
    "Ne": 10,
    "Na": 11,
    "Mg": 12,
    "Al": 13,
    "Si": 14,
    "P": 15,
    "S": 16,
    "Cl": 17,
    "Ar": 18,
    "K": 19,
    "Ca": 20,
    "Sc": 21,
    "Ti": 22,
    "V": 23,
    "Cr": 24,
    "Mn": 25,
    "Fe": 26,
    "Co": 27,
    "Ni": 28,
    "Cu": 29,
    "Zn": 30,
    "Ga": 31,
    "Ge": 32,
    "As": 33,
    "Se": 34,
    "Br": 35,
    "Kr": 36,
    "Rb": 37,
    "Sr": 38,
    "Y": 39,
    "Zr": 40,
    "Nb": 41,
    "Mo": 42,
    "Tc": 43,
    "Ru": 44,
    "Rh": 45,
    "Pd": 46,
    "Ag": 47,
    "Cd": 48,
    "In": 49,
    "Sn": 50,
    "Sb": 51,
    "Te": 52,
    "I": 53,
    "Xe": 54,
    "Cs": 55,
    "Ba": 56,
    "La": 57,
    "Ce": 58,
    "Pr": 59,
    "Nd": 60,
    "Pm": 61,
    "Sm": 62,
    "Eu": 63,
    "Gd": 64,
    "Tb": 65,
    "Dy": 66,
    "Ho": 67,
    "Er": 68,
    "Tm": 69,
    "Yb": 70,
    "Lu": 71,
    "Hf": 72,
    "Ta": 73,
    "W": 74,
    "Re": 75,
    "Os": 76,
    "Ir": 77,
    "Pt": 78,
    "Au": 79,
    "Hg": 80,
    "Tl": 81,
    "Pb": 82,
    "Bi": 83,
    "Th": 90,
    "Pa": 91,
    "U": 92,
    "Np": 93,
    "Pu": 94,
    "Am": 95,
    "Cm": 96,
    "Bk": 97,
    "Cf": 98,
    "Es": 99,
    "Fm": 100,
    "Md": 101,
    "No": 102,
    "Lr": 103,
    "Rf": 104,
    "Db": 105,
    "Sg": 106,
    "Bh": 107,
    "Hs": 108,
    "Mt": 109,
    "Ds": 110,
    "Rg": 111,
    "Cn": 112,
    "Nh": 113,
    "Fl": 114,
    "Mc": 115,
    "Lv": 116,
    "Ts": 117,
    "Og": 118,
}
