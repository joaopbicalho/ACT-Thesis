import os
import pickle
import numpy as np
import cupy as cp
import scipy.integrate as integrate
import pandas as pd
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from matplotlib.pyplot import figure
import scipy.optimize as optimize
from tqdm import tqdm
import joblib

class ACT:
    def __init__(self, FS=256, length=76, dict_addr='dict_cache.p', 
                 tc_info=(0, 38, 2), fc_info=(.7, 15, .2), 
                 logDt_info=(-4, -1, .3), c_info=(-30, 30, 3), 
                 complex=False, force_regenerate=False, mute=False):
        """
        Initializes the Adaptive Chirplet Transform module. If a cached dictionary exists, it is loaded 
        (and converted to GPU arrays); otherwise, the dictionary is generated on the GPU.
        """
        self.FS = FS
        self.length = length
        self.dict_addr = dict_addr

        self.tc_info = tc_info
        self.fc_info = fc_info
        self.logDt_info = logDt_info
        self.c_info = c_info

        self.complex = complex
        self.float32 = True

        if not mute:
            print('\n===============================================')
            print('INITIALIZING ADAPTIVE CHIRPLET TRANSFORM MODULE')
            print('===============================================\n')

        if os.path.exists(self.dict_addr) and not force_regenerate:
            if not mute:
                print("Found Chirplet Dictionary, Loading File...")
            # Load as numpy arrays then convert to CuPy arrays for GPU computations.
            dict_mat_np, param_mat_np = joblib.load(self.dict_addr)
            self.dict_mat = cp.array(dict_mat_np, dtype=cp.float32)
            self.param_mat = cp.array(param_mat_np, dtype=cp.float32)
        else:
            if not mute:
                print("Did not find cached chirplet dictionary matrix, generating chirplet dictionary...\n")
            self.generate_chirplet_dictionary(debug=True)
            if not mute:
                print("\nDone Generating Chirplet Dictionary")
            if not mute:
                print("\nCaching Generated Dictionary/Parameter Matrices...")
            # Save as numpy arrays so that the file is CPU friendly.
            joblib.dump((cp.asnumpy(self.dict_mat), cp.asnumpy(self.param_mat)), self.dict_addr)
            if not mute:
                print("Done Caching.")

        if not mute:
            print("=====================================================")
            print("DONE INITIALIZING ADAPTIVE CHIRPLET TRANSFORM MODULE.")
            print("=====================================================")

    def g(self, tc=0, fc=1, logDt=0, c=0):
        """
        Creates a GPU array of a Gaussian chirplet.
        
        Parameters:
          - tc: Time center (in samples)
          - fc: Frequency center (in Hertz)
          - logDt: Log of Delta_t (chirplet length, where 0 corresponds to one second)
          - c: Chirp rate (in Hertz/sec)
          
        Returns:
          A CuPy array representing the chirplet.
        """
        # Convert tc from samples to seconds
        tc_sec = tc / self.FS

        Dt = cp.exp(logDt)  # Delta_t value
        t = cp.arange(self.length) / self.FS  # Time array on GPU

        gaussian_window = cp.exp(-0.5 * ((t - tc_sec) / Dt) ** 2)
        complex_exp = cp.exp(2j * cp.pi * (c * (t - tc_sec) ** 2 + fc * (t - tc_sec)))

        final_chirplet = gaussian_window * complex_exp

        if not self.complex:
            final_chirplet = cp.real(final_chirplet)

        if self.float32:
            final_chirplet = final_chirplet.astype(cp.float32)

        return final_chirplet

    def generate_chirplet_dictionary(self, debug=False):
        """
        Generates the chirplet dictionary along with its associated parameter matrix.
        Uses CPU iteration over the parameter ranges and GPU operations to compute each chirplet.
        
        Returns:
          A tuple (dict_mat, param_mat) where:
            - dict_mat: 2-D CuPy array of chirplets (each row is one chirplet)
            - param_mat: 2-D CuPy array of corresponding parameters [tc, fc, logDt, c]
        """
        # Iterate on CPU for the parameter values
        tc_vals = np.arange(self.tc_info[0], self.tc_info[1], self.tc_info[2])
        fc_vals = np.arange(self.fc_info[0], self.fc_info[1], self.fc_info[2])
        logDt_vals = np.arange(self.logDt_info[0], self.logDt_info[1], self.logDt_info[2])
        c_vals = np.arange(self.c_info[0], self.c_info[1], self.c_info[2])
        
        dict_size = int(len(tc_vals) * len(fc_vals) * len(logDt_vals) * len(c_vals))
        if debug:
            print("Dictionary length: {}".format(dict_size))

        # Pre-allocate lists to store chirplets and parameters.
        dict_list = []
        param_list = []

        cnt = 0
        slow_cnt = 1
        for tc in tc_vals:
            if debug:
                print('\n{}/{}: \t'.format(slow_cnt, len(tc_vals)), end='')
                slow_cnt += 1
            for fc in fc_vals:
                if debug:
                    print('.', end='')
                for logDt in logDt_vals:
                    for c in c_vals:
                        chirplet = self.g(tc=tc, fc=fc, logDt=logDt, c=c)
                        dict_list.append(chirplet)
                        param_list.append([tc, fc, logDt, c])
                        cnt += 1

        # Stack the list into a 2-D CuPy array.
        self.dict_mat = cp.stack(dict_list, axis=0).astype(cp.float32)
        self.param_mat = cp.array(param_list, dtype=cp.float32)

        return self.dict_mat, self.param_mat

    def search_dictionary(self, signal):
        """
        Searches for the chirplet that gives the maximum projection with the input signal.
        
        Parameters:
          - signal: A CuPy array of the signal (must be on GPU)
          
        Returns:
          - index: The index (int) in the dictionary with the maximum dot product.
          - value: The maximum dot product value (float).
        """
        dot_products = cp.dot(self.dict_mat, signal)
        ind = int(cp.argmax(dot_products).get())
        val = float(cp.max(dot_products).get())
        return ind, val

    def transform(self, signal, order=5, debug=False):
        """
        Performs a P-Order Adaptive Chirplet Transform approximation of the input signal.
        
        Parameters:
          - signal: The input signal (as a NumPy array)
          - order: The approximation order (number of chirplets to use)
          
        Returns:
          A dictionary with:
            - 'params': The optimized parameter sets (NumPy array)
            - 'coeffs': The chirplet coefficients (NumPy array)
            - 'signal': The original signal (NumPy array)
            - 'error': The residual error (float)
            - 'residue': The residue signal (NumPy array)
            - 'approx': The signal approximation (NumPy array)
        """
        # Move the signal to the GPU.
        signal_gpu = cp.asarray(signal, dtype=cp.float32)
        residue = cp.copy(signal_gpu)
        approx = cp.zeros_like(signal_gpu)

        # Pre-allocate arrays to store the parameters and coefficients.
        param_list = np.zeros((order, 4), dtype=np.float32)
        coeff_list = np.zeros(order, dtype=np.float32)

        if debug:
            print('Beginning {}-Order Transform of Input Signal...'.format(order))
        for P in range(order):
            if debug:
                print(".", end="")

            # Search for the best matching chirplet in the dictionary.
            ind, val = self.search_dictionary(residue)

            # Get the coarse parameter estimation (convert from GPU to CPU for the optimizer).
            params_coarse = cp.asnumpy(self.param_mat[ind])

            # Fine-tune parameters using SciPy’s optimizer (which works on CPU arrays).
            res = optimize.minimize(self.minimize_this, params_coarse, args=(cp.asnumpy(residue)))
            new_params = res.x

            if res.status != 0 and debug:
                print('OPTIMIZER DID NOT TERMINATE SUCCESSFULLY!!!')
                print('Message: {}'.format(res.message))

            # Generate the updated chirplet using the optimized parameters.
            updated_base_chirplet = self.g(tc=new_params[0], fc=new_params[1], logDt=new_params[2], c=new_params[3])
            updated_chirplet_coeff = float(cp.dot(updated_base_chirplet, residue).get() / self.FS)

            new_chirp = updated_base_chirplet * updated_chirplet_coeff

            # Update the residue and the current approximation.
            residue = residue - new_chirp
            approx = approx + new_chirp

            param_list[P] = new_params
            coeff_list[P] = updated_chirplet_coeff

        if debug:
            print('')

        result = {
            'params': param_list,
            'coeffs': coeff_list,
            'signal': cp.asnumpy(signal_gpu),
            'error': float(cp.sum(residue).get()),
            'residue': cp.asnumpy(residue),
            'approx': cp.asnumpy(approx)
        }
        return result

    def minimize_this(self, coeffs, signal):
        """
        Cost function used in the fine-tuning step.
        
        Parameters:
          - coeffs: Parameter array [tc, fc, logDt, c] (CPU NumPy array)
          - signal: The signal (CPU NumPy array)
          
        Returns:
          The negative absolute dot product (a scalar) as the cost.
        """
        # Compute the chirplet on GPU using the given parameters.
        atom = self.g(tc=coeffs[0], fc=coeffs[1], logDt=coeffs[2], c=coeffs[3])
        dot_val = cp.dot(atom, cp.asarray(signal, dtype=cp.float32))
        return -1 * float(cp.abs(dot_val).get())
