import numpy as np
def whiten_aux(data_f, data_f_whiten, index, half_width, avarage_window_width, half_width_pos):
     
     for j in index:
         den = np.sum(np.abs(data_f[j:j + 2 * half_width])) / avarage_window_width
         data_f_whiten[j + half_width_pos] = data_f[j + half_width_pos] / den

     return data_f_whiten

def whiten_aux_horizontals(data_f_N, data_f_whiten_N, data_f_E, data_f_whiten_E, index, half_width,
                           avarage_window_width, half_width_pos):
    for j in index:
        den = 0.5*(np.sum(np.abs(data_f_N[j:j + 2 * half_width]) + np.abs(data_f_E[j:j + 2 * half_width]))/avarage_window_width)
        data_f_whiten_N[j + half_width_pos] = data_f_whiten_N[j + half_width_pos] / den
        data_f_whiten_E[j + half_width_pos] = data_f_whiten_E[j + half_width_pos] / den
    return data_f_whiten_N, data_f_whiten_E
