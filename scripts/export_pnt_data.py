import sys, pyreadr
r = pyreadr.read_r(sys.argv[1]); out = sys.argv[2]
it = r["items"][["target", "itemDifficulty", "discrimination"]].rename(columns={"target": "word", "itemDifficulty": "irt_difficulty_t", "discrimination": "irt_discrimination"})
it.to_csv(f"{out}/pnt_irt_raw.csv", index=False)
r["thetas"].rename(columns={"thetas": "theta_t"}).to_csv(f"{out}/pnt_thetas.csv", index=False)
print(len(it), len(r["thetas"]))
