"""Dev CLI: `python -m analysis.games.osu.replay <replay.osr> [chart.osu]`."""
import sys

from analysis.games.osu.replay import parse_replay, find_osu_dirs


def main():
    if len(sys.argv) < 2:
        print("usage: python -m analysis.games.osu.replay <replay.osr> [chart.osu]")
        print(find_osu_dirs())
        return 0
    osr = sys.argv[1]
    osu = sys.argv[2] if len(sys.argv) > 2 else None
    rep = parse_replay(osr, osu_path=osu,
                       songs_dir=find_osu_dirs().get('songs_dir'))
    n = len(rep['offsets'])
    m = int(rep['misses'].sum())
    clean = rep['offsets'][~rep['misses']]
    print(f"keycount: {rep['keycount']}")
    print(f"chart: {rep['chart_meta']}")
    print(f"notes: {n}  hits: {n - m}  misses: {m}")
    if len(clean):
        print(f"mean: {clean.mean()*1000:+.2f}ms  std: {clean.std()*1000:.2f}ms")
    return 0


if __name__ == '__main__':
    sys.exit(main())
