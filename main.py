# AUTHOR : MILO DERVIEUX & RAPHAEL DEBACHE
# DATE : 2024-03-28
# DESCRIPTION : TP Multi-Antenna Systems - Direction of Arrival Estimation
#               (Capon and MUSIC Methods)


# IMPORTS
import numpy as np
from numpy.typing import NDArray
import matplotlib.pyplot as plt

def a(M, theta):
    """Steering vector function for M sensors and angle theta"""
    return np.exp(1j * np.pi * np.sin(theta*2*np.pi/360) * np.arange(M))

def A(M, theta):
    """Construct steering matrix for multiple angles"""
    A_matrix = np.zeros((M, len(theta)), dtype=complex)
    for i in range(len(theta)):
        A_matrix[:, i] = a(M, theta[i])
    return A_matrix

def generate_signal(M, N, theta, sigma2_s, sigma2_n):
    """Generate synthetic signal received by M sensors for N samples
    with signal powers sigma2_s and noise power sigma2_n"""
    A_matrix = A(M, theta)
    y = np.zeros((M, N), dtype=complex)
    for n in range(N):
        s_n = np.sqrt(np.diag(sigma2_s)) @ (np.random.randn(K) + 1j * np.random.randn(K)) / np.sqrt(2)
        v_n = np.sqrt(sigma2_n) * (np.random.randn(M) + 1j * np.random.randn(M)) / np.sqrt(2)
        y[:, n] = A_matrix @ s_n + v_n
    return y

def cov_empirique(Y):
    """Empirical covariance matrix estimation from data Y"""
    N = Y.shape[1]
    R = np.zeros((Y.shape[0], Y.shape[0]), dtype=Y.dtype)
    for n in range(N):
        R += np.outer(Y[:, n], np.conj(Y[:, n]))
    return R / N

def w_capon(R: NDArray[np.complex128], M: int, theta: float) -> NDArray[np.complex128]:
    """
    M is the number of sensors
    R is the covariance matrix of noise for MVDR or the covariance matrix of signal for CAPON
    theta is the angle of arrival in degree 

    return beamforming filter
    """
    a_theta = a(M, theta)
    R_inv = np.linalg.inv(R)
    numerator = R_inv @ a_theta
    denominator = np.conj(a_theta).T @ R_inv @ a_theta
    return numerator / denominator

def boucle_sigma2_n(M, N, theta, sigma2_s, step):
    plt.figure()
    for sigma2_n in range(10, 50, 10):
        Y = generate_signal(M, N, theta, sigma2_s, sigma2_n*step)
        # Capon filter
        theta_scan = np.linspace(-90, 90, 181)
        P_capon = np.zeros(len(theta_scan), dtype=complex)  # Power spectrum initialization
        R_emp = cov_empirique(Y)
        R_inv = np.linalg.inv(R_emp)
        for i, angle in enumerate(theta_scan):
            P_capon[i] = 1 / (np.conj(a(M, angle)).T @ R_inv @ a(M, angle))

        # Power spectrum display
        plt.plot(theta_scan, 10 * np.log10(np.abs(P_capon) / np.max(np.abs(P_capon))),
                 label=f"$\\sigma_v^2 = {sigma2_n*step}$")
    plt.title(f"Power Spectrum - Capon Filter ({M} sensors)")
    plt.xlabel("Angle (degrees)")
    plt.ylabel("Power (dB)")
    plt.legend(title="Noise variance", bbox_to_anchor=(1.05, 1), loc="upper left")
    plt.subplots_adjust(right=0.8)
    plt.grid()

def boucle_sigma2_music(M, N, theta, sigma2_s, step):
    plt.figure()
    for sigma2_n in range(350, 400, 10):
        Y = generate_signal(M, N, theta, sigma2_s, sigma2_n*step)
        R_emp = cov_empirique(Y)
        
        # MUSIC filter  
        theta_scan = np.linspace(-90, 90, 181)
        P_music = np.zeros(len(theta_scan), dtype=complex)
        val_propre, vect_propre = np.linalg.eigh(R_emp)
        K = len(theta)
        proj_sig = vect_propre[:, -K:] @ np.conj(vect_propre[:, -K:]).T
        proj_noise = np.eye(M) - proj_sig
        A_theta = A(M, theta_scan)
        dist = np.linalg.norm(proj_noise @ A_theta, axis=0)**2
        P_music = 1 / dist
        plt.plot(theta_scan, 10 * np.log10(np.abs(P_music) / np.max(np.abs(P_music))),
                 label=f"$\\sigma_v^2 = {sigma2_n*step}$")
    plt.title(f"Power Spectrum - MUSIC Filter ({M} sensors)")
    plt.xlabel("Angle (degrees)")
    plt.ylabel("Power (dB)")
    plt.legend(title="Noise variance", bbox_to_anchor=(1.05, 1), loc="upper left")
    plt.subplots_adjust(right=0.8)
    plt.grid()


def part2(filename):
    """Given a .mat file containing multiple recordings"""

    from scipy.io import loadmat
    from scipy.signal import hilbert, find_peaks
    import sounddevice as sd   
    import soundfile as sf
    """-------------> pip install sounddevice"""

    data = loadmat(filename)
    Y = data['data']  
    Fs = data['Fs'][0,0]

    Fc = 1635  # Center frequency (Hz)

    Y_postHilbert = hilbert(Y)  

    Y_noise = Y_postHilbert[:, 0:2*Fs]
    Y_signal = Y_postHilbert[:, 2*Fs::]  # Signal after 2 seconds of noise only

    sigma2_n = abs(cov_empirique(Y_noise).diagonal().mean())  # Estimated noise power
    print(f"Estimated noise power: {sigma2_n}")  

    M = np.shape(Y)[0]  # Number of microphones
    R_emp = cov_empirique(Y_signal)
    [R_vp, Vec_R_vp] = np.linalg.eig(R_emp)

    K = 5  # Number of detected sources

    proj_sig = Vec_R_vp[:, :K] @ np.conj(Vec_R_vp[:, :K]).T
    proj_noise = np.eye(M) - proj_sig

    theta_scan = np.linspace(-90, 90, 181)
    dist = np.linalg.norm(proj_noise @ A(M, theta_scan), axis=0)**2
    P_music = 1 / dist

    P_music_db = 10 * np.log10(np.abs(P_music) / np.max(np.abs(P_music)))

    # Selecting sources
    ind_sources, _ = find_peaks(P_music_db, height=-20)
    theta_sources = theta_scan[ind_sources]

    # MVDR Beamforming 
    R_emp_noise = cov_empirique(Y_noise)
    Voice_number = 4

    # MVDR Beamformer filter
    w_MVDR = w_capon(R_emp_noise, M, theta_sources[Voice_number])
    Voice0_MVDR = np.conj(w_MVDR).T @ Y_signal

    # CAPON Beamforming
    w_CAPON = w_capon(R_emp, M, theta_sources[Voice_number])
    Voice0_CAPON = np.conj(w_CAPON).T @ Y_signal

    # Eigenvalue histogram
    plt.figure()
    plt.hist(np.real(R_vp), density=True, bins=30, color='blue', edgecolor='black')
    plt.title("Histogram of Eigenvalues of R_emp | Signal Part 2")
    plt.xlabel("Eigenvalues")
    plt.ylabel("Frequency")
    plt.grid()

    # MUSIC power spectrum display
    plt.figure()
    plt.plot(theta_scan, P_music_db)
    plt.plot(theta_scan[ind_sources],  P_music_db[ind_sources], "x")
    plt.title("Power Spectrum - MUSIC Method | Signal Part 2")
    plt.xlabel("Angle (degrees)")
    plt.ylabel("Power (dB)")
    plt.grid()

    frequencies = np.fft.fftshift(np.fft.fftfreq(len(Voice0_CAPON), 1/Fs))
    Pxx = np.fft.fftshift(np.abs(np.fft.fft(Voice0_CAPON))**2)
    plt.plot(frequencies, 10 * np.log10(Pxx + 1e-12))
    plt.xlabel("Frequency (Hz)")
    plt.ylabel("Power (dB)")
    plt.title("Power Spectrum")
    plt.grid()
    plt.show()

    frequencies = np.fft.fftshift(np.fft.fftfreq(len(Voice0_MVDR), 1/Fs))
    Pxx = np.fft.fftshift(np.abs(np.fft.fft(Voice0_MVDR))**2)
    plt.plot(frequencies, 10 * np.log10(Pxx + 1e-12))
    plt.xlabel("Frequency (Hz)")
    plt.ylabel("Power (dB)")
    plt.title("Power Spectrum")
    plt.grid()
    plt.show()

    # Extracting and normalizing audio signals
    signal_audio = np.real(Y_signal[0])
    signal_audio /= np.max(np.abs(signal_audio))
    sf.write('Observed_signal_0.wav', signal_audio, Fs)

    signal_audio1 = np.real(Y_signal[1])
    signal_audio1 /= np.max(np.abs(signal_audio1))  
    sf.write('Observed_signal_1.wav', signal_audio1, Fs)


if __name__ == "__main__":

    PLOT = True
    ONLY_P2 = False

    if not ONLY_P2:
        # Parameters
        M = 14  # Number of sensors
        N = 500  # Number of samples
        theta = [40, 45]  # Signal arrival angles
        sigma2_s = [1, 1]  # Signal power
        K = len(theta)  # Number of sources
        sigma2_n = 0.1  # Noise power
        
        # Generate signal
        Y = generate_signal(M, N, theta, sigma2_s, sigma2_n)

        # Capon filter
        theta_scan = np.linspace(-90, 90, 181)
        P_capon = np.zeros(len(theta_scan), dtype=complex)
        R_emp = cov_empirique(Y)
        R_inv = np.linalg.inv(R_emp)
        for i, angle in enumerate(theta_scan):
            P_capon[i] = 1 / (np.conj(a(M, angle)).T @ R_inv @ a(M, angle))

        if PLOT:
            fig = plt.figure()
            ax = fig.add_subplot(111)
            ax.plot(theta_scan, 10 * np.log10(np.abs(P_capon) / np.max(np.abs(P_capon))))
            ax.set_title("Power Spectrum - Capon Filter")
            ax.set_xlabel("Angle (degrees)")
            ax.set_ylabel("Power (dB)")
            ax.grid()

            boucle_sigma2_n(M, N, theta, sigma2_s, 0.01)
            boucle_sigma2_music(M, N, theta, sigma2_s, 0.01)

        [R_vp, Vec_R_vp] = np.linalg.eig(R_emp)

        Seuil = 5 
        K = len(theta)

        proj_sig = Vec_R_vp[:, :K] @ np.conj(Vec_R_vp[:, :K]).T
        proj_noise = np.eye(M) - proj_sig

        theta_scan = np.linspace(-90, 90, 181)
        dist = np.linalg.norm(proj_noise @ A(M, theta_scan), axis=0)**2
        P_music = 1 / dist

        if PLOT:
            plt.figure()
            plt.hist(np.real(R_vp), density=True, bins=30, alpha=0.7, color='blue', edgecolor='black')
            plt.title("Histogram of Eigenvalues of R_emp")
            plt.xlabel("Eigenvalues")
            plt.ylabel("Frequency")
            plt.grid()

            plt.figure()
            plt.plot(theta_scan * 360 / (2 * np.pi), 10 * np.log10(np.abs(P_music) / np.max(np.abs(P_music))))
            plt.title("Power Spectrum - MUSIC Method")
            plt.xlabel("Angle (degrees)")
            plt.ylabel("Power (dB)")
            plt.grid()

            plt.figure()
            plt.plot(theta_scan, 10 * np.log10(np.abs(P_capon) / np.max(np.abs(P_capon))), label="Capon")
            plt.plot(theta_scan, 10 * np.log10(np.abs(P_music) / np.max(np.abs(P_music))), label="MUSIC")
            plt.title(f"Comparison of Power Spectra - {M} sensors")
            plt.xlabel("Angle (degrees)")
            plt.ylabel("Power (dB)")
            plt.legend()
            plt.grid()

    # part2("data.mat")

    plt.show()
