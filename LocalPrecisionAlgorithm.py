import numpy as np
import pandas as pd
from scipy.optimize import curve_fit

def def_roi_from_df(df):
    roi_frame = np.zeros([4, 2])
    roi_frame[0, 0] = df.iloc[:, 0].min()  # Xmin
    roi_frame[0, 1] = df.iloc[:, 0].max()  # Xmax
    roi_frame[1, 0] = df.iloc[:, 1].min()  # Ymin
    roi_frame[1, 1] = df.iloc[:, 1].max()  # Ymax
    roi_frame[2, 0] = df.iloc[:, 2].min()  # Tmin
    roi_frame[2, 1] = df.iloc[:, 2].max()  # Tmax
    roi_frame[3, 0] = df.iloc[:, 3].min()  # Imin
    roi_frame[3, 1] = df.iloc[:, 3].max()  # Imax
    return roi_frame

def print_roi(roi_frame):
    print('Xmin [nm]:     ' + str(int(roi_frame[0, 0])))
    print('Xmax [nm]:     ' + str(int(roi_frame[0, 1])))
    print('Ymin [nm]:     ' + str(int(roi_frame[1, 0])))
    print('Ymax [nm]:     ' + str(int(roi_frame[1, 1])))
    print('Tmin [frame]:  ' + str(int(roi_frame[2, 0])))
    print('Tmax [frame]:  ' + str(int(roi_frame[2, 1])))
    print('Imin [counts]: ' + str(int(roi_frame[3, 0])))
    print('Imax [counts]: ' + str(int(roi_frame[3, 1])))

def ROI(locs, roi_frame):
    idx1 = locs[:, 0] >= roi_frame[0, 0]
    idx2 = locs[:, 0] <= roi_frame[0, 1]
    idy1 = locs[:, 1] >= roi_frame[1, 0]
    idy2 = locs[:, 1] <= roi_frame[1, 1]
    idt1 = locs[:, 2] >= roi_frame[2, 0]
    idt2 = locs[:, 2] <= roi_frame[2, 1]
    idi1 = locs[:, 3] >= roi_frame[3, 0]
    idi2 = locs[:, 3] <= roi_frame[3, 1]
    locs1 = locs[(idx1 & idx2 & idy1 & idy2 & idt1 & idt2 & idi1 & idi2), :]
    return locs1

def N_N(array, value):
    idx = (np.abs(array - value)).argmin()
    return array[idx]

def Acc_Calculator(Nlocs, MaxFrame):
    print('Estimating coordinate-based localization precision...')
    leftMax = np.searchsorted(Nlocs[:, 2], MaxFrame, side='left')
    lengthlocs = leftMax - 1
    NearNeigh = np.zeros([lengthlocs, 1])    
    for i in range(lengthlocs):
        loc = Nlocs[i, :]
        j = 1 + loc[2]
        if i == 0 or loc[2] > Nlocs[(i-1), 2]:
            left = np.searchsorted(Nlocs[:, 2], j, side='left')
            right = np.searchsorted(Nlocs[:, 2], j, side='right')
            Frlocs = Nlocs[left:right, :]
        if left == right:
            NearNeigh[i] = 0
        else:
            Dist = np.sqrt((loc[0] - Frlocs[:, 0])**2 + (loc[1] - Frlocs[:, 1])**2)
            NearNeigh[i] = N_N(Dist, 0)
            if NearNeigh[i] > 200:
                NearNeigh[i] = 200
    print('Done!')
    return NearNeigh

def Area(r, y):
    return abs(np.trapz(y, r))

def CFunc2dCorr(r, a, rc, w, F, A, O):
    return (r / (2 * a**2)) * np.exp(-r**2 / (4 * a**2)) * A + \
           (F / (w * np.sqrt(2 * np.pi))) * np.exp(-0.5 * ((r - rc) / w)**2) + O * r

def CFit_resultsCorr(r, y):
    A = Area(r, y)
    p0 = np.array([10.0, 201, 100, A / 2, A / 2, y[98] / 200])
    popt, pcov = curve_fit(CFunc2dCorr, r, y, p0)
    print('Estimated localization precision at:')
    print(popt)
    return popt, pcov

def Loc_Acc(df):
    roi_frame = def_roi_from_df(df)
    print_roi(roi_frame)
    locs = df.to_numpy()  # Convert DataFrame to numpy array
    roi_locs = ROI(locs, roi_frame)
    NearNeigh = Acc_Calculator(roi_locs, roi_frame[2, 1])
    ahist = np.histogram(NearNeigh, bins=99, range=(1, 199), density=False)
    ar = ahist[1][1:] - 1
    ay = ahist[0]
    aF, aFerr = CFit_resultsCorr(ar, ay)
    ayf = CFunc2dCorr(ar, *aF)
    return aF[0], np.sqrt(aFerr[0,0])
    

def get_local_precision(df):
    def_roi_from_df(df)
    print_roi(def_roi_from_df(df))
    return Loc_Acc(df)

if __name__ == "__main__":
    df = pd.read_csv('Total_Localization_file_initON_Spatial_Radius_2_pixel_init_OFF.txt', delimiter=' ', header=1)
    print(get_local_precision(df))
