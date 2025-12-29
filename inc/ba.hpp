#pragma once

/**
 * Bundle Adjustment (BA) Module
 * 
 * This module contains the implementation of bundle adjustment for DPVO.
 * The bundleAdjustment() function is a member function of the DPVO class.
 * 
 * Implementation file: app/src/ba.cpp
 * Declaration: app/inc/dpvo.hpp (as private member function)
 * 
 * The BA function solves the Schur-complement system for optimizing
 * camera poses and inverse depths based on reprojection residuals.
 */

