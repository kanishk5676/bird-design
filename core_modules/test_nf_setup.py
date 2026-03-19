#!/usr/bin/env python3
"""
Quick test script for NeuralFoil analysis
Tests on a small sample of airfoils to verify installation and setup
"""

import sys
from pathlib import Path

# Add core_modules to path
sys.path.insert(0, str(Path(__file__).parent / "core_modules"))

def test_imports():
    """Test all required imports"""
    print("Testing imports...")
    try:
        import numpy as np
        print("  ✓ numpy")
    except ImportError:
        print("  ✗ numpy - Install: pip install numpy")
        return False
        
    try:
        import pandas as pd
        print("  ✓ pandas")
    except ImportError:
        print("  ✗ pandas - Install: pip install pandas")
        return False
        
    try:
        import matplotlib.pyplot as plt
        print("  ✓ matplotlib")
    except ImportError:
        print("  ✗ matplotlib - Install: pip install matplotlib")
        return False
        
    try:
        import seaborn as sns
        print("  ✓ seaborn")
    except ImportError:
        print("  ✗ seaborn - Install: pip install seaborn")
        return False
        
    try:
        from tqdm import tqdm
        print("  ✓ tqdm")
    except ImportError:
        print("  ✗ tqdm - Install: pip install tqdm")
        return False
        
    try:
        import neuralfoil as nf
        print("  ✓ neuralfoil")
    except ImportError:
        print("  ✗ neuralfoil - Install: pip install neuralfoil")
        print("\n  NeuralFoil Installation:")
        print("    pip install neuralfoil")
        print("  See docs/NEURALFOIL_SETUP.md for details")
        return False
    
    return True


def test_airfoil_loading():
    """Test that we can load airfoil files"""
    print("\nTesting airfoil file access...")
    
    airfoil_dir = Path(__file__).parent / "OUTPUT" / "airfoils"
    
    if not airfoil_dir.exists():
        print(f"  ✗ Airfoil directory not found: {airfoil_dir}")
        return False
    
    # Find a sample .dat file
    sample_files = list(airfoil_dir.glob("**/*.dat"))
    
    if not sample_files:
        print("  ✗ No .dat files found")
        return False
    
    print(f"  ✓ Found {len(sample_files)} airfoil files")
    
    # Try loading one
    sample = sample_files[0]
    print(f"  Testing load: {sample.name}")
    
    try:
        import numpy as np
        coords = []
        with open(sample, 'r') as f:
            for line in f:
                line = line.strip()
                if line.startswith('#') or not line:
                    continue
                parts = line.split()
                if len(parts) == 2:
                    try:
                        coords.append([float(parts[0]), float(parts[1])])
                    except ValueError:
                        continue
        
        if len(coords) < 10:
            print(f"  ✗ Too few coordinates: {len(coords)}")
            return False
        
        print(f"  ✓ Loaded {len(coords)} coordinate points")
        
    except Exception as e:
        print(f"  ✗ Failed to load: {e}")
        return False
    
    return True


def test_neuralfoil_simulation():
    """Run a quick simulation on one airfoil"""
    print("\nTesting NeuralFoil simulation...")
    
    try:
        import neuralfoil as nf
        import numpy as np
        
        # Simple test airfoil (NACA 0012-like)
        n = 50
        x = np.linspace(0, 1, n)
        y_upper = 0.12 * (0.2969*np.sqrt(x) - 0.1260*x - 0.3516*x**2 + 0.2843*x**3 - 0.1015*x**4)
        y_lower = -y_upper
        
        # Combine upper and lower surfaces
        coords = np.vstack([
            np.column_stack([x[::-1], y_upper[::-1]]),
            np.column_stack([x[1:], y_lower[1:]])
        ])
        
        print("  Running simulation at Re=1e5, alpha=4°...")
        result = nf.get_aero_from_coordinates(
            coordinates=coords,
            alpha=4.0,
            Re=1e5
        )
        
        # NeuralFoil returns arrays, extract first value
        cl_raw = result.get('CL', [0])
        cd_raw = result.get('CD', [1e-6])
        cl = float(cl_raw[0]) if hasattr(cl_raw, '__iter__') else float(cl_raw)
        cd = float(cd_raw[0]) if hasattr(cd_raw, '__iter__') else float(cd_raw)
        
        print(f"  ✓ Simulation successful")
        print(f"    CL = {cl:.4f}")
        print(f"    CD = {cd:.5f}")
        print(f"    L/D = {cl/cd:.2f}")
        
        return True
        
    except Exception as e:
        print(f"  ✗ Simulation failed: {e}")
        return False


def main():
    print("="*70)
    print("NeuralFoil Analysis - Quick Test")
    print("="*70)
    
    tests_passed = 0
    tests_total = 3
    
    # Test 1: Imports
    if test_imports():
        tests_passed += 1
    else:
        print("\n✗ Import test failed")
        print("Install missing packages and try again")
        return False
    
    # Test 2: Airfoil files
    if test_airfoil_loading():
        tests_passed += 1
    else:
        print("\n✗ Airfoil loading test failed")
        return False
    
    # Test 3: NeuralFoil simulation
    if test_neuralfoil_simulation():
        tests_passed += 1
    else:
        print("\n✗ NeuralFoil simulation test failed")
        return False
    
    print("\n" + "="*70)
    print(f"✓ All tests passed ({tests_passed}/{tests_total})")
    print("="*70)
    print("\nYour system is ready for NeuralFoil analysis!")
    print("\nTo run full analysis:")
    print("  python core_modules/neuralfoil.py")
    print("\nEstimated runtime: 20-30 minutes for all 10,975 airfoils")
    print("\nFor more info, see: docs/NEURALFOIL_SETUP.md")
    print("="*70)
    
    return True


if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)
