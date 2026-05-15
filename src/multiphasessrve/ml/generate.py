"""CLI do generowania mikrostruktur SSRVE z wytrenowanego modelu.

Przyklad uzycia:

    uv run python -m multiphasessrve.ml.generate --cr 10 --n 10 --out ./output_cr10

Generuje 10 mikrostruktur dla CR=10 i zapisuje do katalogu output_cr10/:
  - prediction.csv     -- deterministyczna predykcja z glowic modelu
  - parameters.csv     -- per-sample statystyki po wzroscie Voronoi
  - ssrve_00.png ...   -- obrazki mikrostruktur
  - starting_points_00.csv ... -- punkty zarodkowania
"""

from __future__ import annotations

import argparse
from pathlib import Path

from multiphasessrve.ml.config import CVAEConfig
from multiphasessrve.ml.evaluate import load_model
from multiphasessrve.ml.inference import predict, generate_ssrves, PHASE_NAMES


def main():
    parser = argparse.ArgumentParser(
        description="Generowanie mikrostruktur SSRVE z modelu ML",
    )
    parser.add_argument("--cr", type=float, required=True,
                        help="Predkosc chlodzenia [stopnie C/s]")
    parser.add_argument("--n", type=int, default=10,
                        help="Liczba mikrostruktur do wygenerowania (default: 10)")
    parser.add_argument("--out", type=str, default=None,
                        help="Katalog wyjsciowy (default: output_cr{CR})")
    parser.add_argument("--checkpoint", type=str, default=None,
                        help="Sciezka do checkpointu (default: checkpoints/best_model.pt)")
    parser.add_argument("--device", type=str, default="cuda",
                        help="Urzadzenie (default: cuda)")
    parser.add_argument("--seed", type=int, default=42,
                        help="Ziarno losowe (default: 42)")
    parser.add_argument("--candidates", type=int, default=20,
                        help="Liczba kandydatow do wygenerowania przed selekcja (default: 20)")
    args = parser.parse_args()

    cfg = CVAEConfig(device=args.device)
    ckpt = args.checkpoint or str(Path(cfg.checkpoint_dir) / "best_model.pt")
    model = load_model(ckpt, cfg)

    out_dir = args.out or f"output_cr{args.cr}"

    # 1. Deterministyczna predykcja z glowic
    pred = predict(model, args.cr, cfg)
    print(f"Predykcja modelu dla CR = {args.cr}:")
    print(f"  Frakcje: F={pred['ferrite_fraction']:.4f}  B={pred['bainite_fraction']:.4f}  "
          f"M={pred['martensite_fraction']:.4f}  P={pred['pearlite_fraction']:.4f}")
    print(f"  Wielkosc ziarna ferrytu: {pred['ferrite_grain_size_um']:.3f} um")
    print(f"  Liczba aktywnych zarodkow: {pred['n_active']}")

    # 2. Generacja N mikrostruktur
    print(f"\nGenerowanie {args.n} mikrostruktur (z {args.candidates} kandydatow, seed={args.seed})...")
    results = generate_ssrves(
        model, args.cr, cfg,
        n=args.n,
        out_dir=out_dir,
        seed=args.seed,
        n_candidates=args.candidates,
    )

    # 3. Podsumowanie
    print(f"\nWyniki ({args.n} najlepszych z {args.candidates} kandydatow):")
    print(f"{'#':>3} {'n_act':>5} {'F':>7} {'B':>7} {'M':>7} {'P':>7} {'GS':>7} {'err':>7}")
    for i, r in enumerate(results):
        pf = r["predicted_fractions"]
        print(f"{i:>3d} {r['n_active']:>5d} "
              f"{pf.get('ferrite', 0):.4f} {pf.get('bainite', 0):.4f} "
              f"{pf.get('martensite', 0):.4f} {pf.get('pearlite', 0):.4f} "
              f"{r['predicted_grain_size']:>7.3f} {r['mean_fraction_error']:>7.4f}")

    print(f"\nPliki zapisane w: {Path(out_dir).resolve()}")
    print(f"  prediction.csv     -- predykcja deterministyczna z glowic")
    print(f"  parameters.csv     -- statystyki per mikrostruktura")
    print(f"  ssrve_*.png        -- obrazki")
    print(f"  starting_points_*.csv -- punkty zarodkowania")


if __name__ == "__main__":
    main()
