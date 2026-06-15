#!/usr/bin/env python3
"""Run the solution on both villages and produce predictions."""

from solve import solve_village

if __name__ == '__main__':
    print("=" * 60)
    print("VADNERBHAIRAV (Nashik)")
    print("=" * 60)
    solve_village('data/34855_vadnerbhairav_chandavad_nashik')
    
    print("\n" + "=" * 60)
    print("MALATAVADI (Kolhapur)")
    print("=" * 60)
    solve_village('data/12429_malatavadi_chandgad_kolhapur')
    
    print("\n" + "=" * 60)
    print("DONE — predictions written for both villages.")
    print("=" * 60)
