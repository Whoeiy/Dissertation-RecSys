# -*- coding: utf-8 -*-
"""
Created on Fri May  7 16:37:14 2021

@author: JenniferYu
"""

from __future__ import (absolute_import, division, print_function,
                        unicode_literals)

cimport numpy as np  # noqa
import numpy as np

from six.moves import range
from six import iteritems


def cosine(n_x, yr, min_support):
    """Compute the cosine similarity between all pairs of users (or items).
    Only **common** users (or items) are taken into account. The cosine
    similarity is defined as:
    .. math::
        \\text{cosine_sim}(u, v) = \\frac{
        \\sum\\limits_{i \in I_{uv}} r_{ui} \cdot r_{vi}}
        {\\sqrt{\\sum\\limits_{i \in I_{uv}} r_{ui}^2} \cdot
        \\sqrt{\\sum\\limits_{i \in I_{uv}} r_{vi}^2}
        }
    or
    .. math::
        \\text{cosine_sim}(i, j) = \\frac{
        \\sum\\limits_{u \in U_{ij}} r_{ui} \cdot r_{uj}}
        {\\sqrt{\\sum\\limits_{u \in U_{ij}} r_{ui}^2} \cdot
        \\sqrt{\\sum\\limits_{u \in U_{ij}} r_{uj}^2}
        }
    depending on the ``user_based`` field of ``sim_options`` (see
    :ref:`similarity_measures_configuration`).
    For details on cosine similarity, see on `Wikipedia
    <https://en.wikipedia.org/wiki/Cosine_similarity#Definition>`__.
    """

    # sum (r_xy * r_x'y) for common ys
    cdef np.ndarray[np.double_t, ndim=2] prods
    # number of common ys
    cdef np.ndarray[np.int_t, ndim=2] freq
    # sum (r_xy ^ 2) for common ys
    cdef np.ndarray[np.double_t, ndim=2] sqi
    # sum (r_x'y ^ 2) for common ys
    cdef np.ndarray[np.double_t, ndim=2] sqj
    # the similarity matrix
    cdef np.ndarray[np.double_t, ndim=2] sim

    cdef int xi, xj
    cdef double ri, rj
    cdef int min_sprt = min_support

    prods = np.zeros((n_x, n_x), np.double)
    freq = np.zeros((n_x, n_x), np.int)
    sqi = np.zeros((n_x, n_x), np.double)
    sqj = np.zeros((n_x, n_x), np.double)
    sim = np.zeros((n_x, n_x), np.double)

    for y, y_ratings in iteritems(yr):
        for xi, ri in y_ratings:
            for xj, rj in y_ratings:
                freq[xi, xj] += 1
                prods[xi, xj] += ri * rj
                sqi[xi, xj] += ri**2
                sqj[xi, xj] += rj**2

    for xi in range(n_x):
        sim[xi, xi] = 1
        for xj in range(xi + 1, n_x):
            if freq[xi, xj] < min_sprt:
                sim[xi, xj] = 0
            else:
                denum = np.sqrt(sqi[xi, xj] * sqj[xi, xj])
                sim[xi, xj] = prods[xi, xj] / denum

            sim[xj, xi] = sim[xi, xj]

    return sim