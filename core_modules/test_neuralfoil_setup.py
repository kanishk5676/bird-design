"""
Quick test to verify NeuralFoil setup works with current airfoils
"""
from pathlib import Path
import neuralfoil as nf
from neuralfoil_analysis import load_dat_file, normalise_airfoil_coordinates, AIRFOIL_DIR

print("=" * 72)
print("NEURALFOIL SETUP TEST")
print("=" * 72)

# Test 1: Find airfoil files
print("\n1. Finding airfoil files...")
dat_files = list(AIRFOIL_DIR.rglob("*.dat"))
print(f"   ✓ Found {len(dat_files)} .dat files")

# Test 2: Load a sample airfoil
print("\n2. Testing .dat file loading...")
if dat_files:
    sample_file = dat_files[0]
    print(f"   Sample: {sample_file.name}")
    try:
        coords, meta = load_dat_file(str(sample_file))
        print(f"   ✓ Loaded {len(coords)} coordinate points")
        print(f"   ✓ Species: {meta.get('Species', 'N/A')}")
        print(f"   ✓ Category: {meta.get('Flight Category', 'N/A')}")
        print(f"   ✓ Estimated Re: {meta.get('_estimated_re', 'N/A'):.2e}" if '_estimated_re' in meta else "   ⚠ No Reynolds in header (will estimate)")
        print(f"   ✓ Chord normalized: x ∈ [{coords[:, 0].min():.3f}, {coords[:, 0].max():.3f}]")
    except Exception as e:
        print(f"   ✗ Error: {e}")
else:
    print("   ✗ No files found")

# Test 3: Run single NeuralFoil simulation
print("\n3. Testing NeuralFoil simulation...")
if dat_files and coords is not None:
    try:
        print(f"   Running: α=4°, Re=1e5...")
        aero = nf.get_aero_from_coordinates(
            coordinates=coords,
            alpha=4,
            Re=1e5,
            model_size="large"  # Use smaller model for quick test
        )
        
        # Debug: print what we got
        print(f"   Aero result type: {type(aero)}")
        print(f"   Aero keys: {list(aero.keys()) if isinstance(aero, dict) else 'Not a dict'}")
        
        # Extract values carefully
        CL = float(aero['CL']) if hasattr(aero['CL'], 'item') else float(aero['CL'])
        CD = float(aero['CD']) if hasattr(aero['CD'], 'item') else float(aero['CD'])
        
        print(f"   ✓ CL = {CL:.4f}")
        print(f"   ✓ CD = {CD:.6f}")
        print(f"   ✓ L/D = {CL/CD:.2f}")
        
        if 'analysis_confidence' in aero:
            conf = float(aero['analysis_confidence']) if hasattr(aero['analysis_confidence'], 'item') else float(aero['analysis_confidence'])
            print(f"   ✓ Confidence = {conf:.3f}")
        
        print(f"   SUCCESS: NeuralFoil is working!")
    except Exception as e:
        import traceback
        print(f"   ✗ NeuralFoil error: {e}")
        print(f"   Traceback:")
        traceback.print_exc()
        print(f"   This might be a missing dependency - try: pip install neuralfoil")

# Test 4: Check categories
print("\n4. Checking category distribution...")
categories = {}
for f in dat_files[:100]:  # Sample first 100 files
    cat = f.parent.name
    categories[cat] = categories.get(cat, 0) + 1

for cat, count in sorted(categories.items()):
    print(f"   {cat:15s}: {count:4d} files")

print("\n" + "=" * 72)
print("CONFIGURATION")
print("=" * 72)
print(f"Airfoil directory : {AIRFOIL_DIR}")
print(f"Total airfoils    : {len(dat_files)}")
print(f"Test matrix       : 13 alphas × 6 Reynolds = 78 conditions per airfoil")
print(f"Total simulations : {len(dat_files) * 78:,}")
print(f"Est. time (5s/sim): ~{len(dat_files) * 78 * 5 / 3600:.1f} hours")
print("=" * 72)

print("\n✓ Setup test complete!")
print("\nTo run full analysis:")
print("  python neuralfoil_analysis.py")
print("\nOr run in background and save output:")
print("  nohup python neuralfoil_analysis.py > neuralfoil.log 2>&1 &")
