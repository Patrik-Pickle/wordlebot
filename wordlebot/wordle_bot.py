#!/usr/bin/env python3
"""Wordle bot: picks the guess that splits the remaining candidates most evenly (max entropy).

  python wordle_bot.py play            # you type the colours Wordle shows you
  python wordle_bot.py solve crane     # watch the bot solve a known answer
  python wordle_bot.py sim             # run every answer, print stats

Feedback format: 5 chars, g = green, y = yellow, b = gray/black   (e.g. "bygbb")
"""
import argparse
from collections import Counter
from pathlib import Path

import numpy as np

HERE = Path(__file__).parent
EMOJI = {0: "⬛", 1: "🟨", 2: "🟩"}


def read_words(path):
    return [w.strip().lower() for w in list(open(path))[1:] if len(w.strip()) == 5]  # [1:] skips the header row


def encode(words):
    return np.array([[ord(c) - 97 for c in w] for w in words], dtype=np.int8)


def pattern_vs_all(g, A, Acnt):
    """Wordle feedback of guess g (5,) against every answer row in A (M,5), as base-3 ints."""
    green = A == g
    rem = Acnt.copy()                      # letters in each answer not yet "used up"
    for i in range(5):
        rem[green[:, i], g[i]] -= 1        # greens consume their letter first
    pat = np.zeros(len(A), dtype=np.int32)
    for i in range(5):
        yellow = ~green[:, i] & (rem[:, g[i]] > 0)
        rem[yellow, g[i]] -= 1             # then yellows, left to right (handles duplicates)
        pat += (green[:, i] * 2 + yellow) * 3 ** i
    return pat.astype(np.uint8)


def parse_feedback(s):
    s = s.strip().lower()
    if len(s) != 5 or any(c not in "gyb" for c in s):
        raise ValueError("use 5 chars from g/y/b, e.g. bygbb")
    return sum({"b": 0, "y": 1, "g": 2}[c] * 3 ** i for i, c in enumerate(s))


def show(word, pat):
    return "".join(EMOJI[(pat // 3 ** i) % 3] for i in range(5)) + "  " + word


class Bot:
    def __init__(self, answers, allowed):
        self.answers = answers
        # guesses = answers first, then the rest, so answer j is also guess j
        self.guesses = answers + [w for w in allowed if w not in set(answers)]
        G, A = encode(self.guesses), encode(answers)
        Acnt = np.zeros((len(answers), 26), dtype=np.int8)
        for i in range(5):
            np.add.at(Acnt, (np.arange(len(answers)), A[:, i]), 1)
        # P[g, a] = feedback if you guess g and the answer is a (computed once)
        self.P = np.stack([pattern_vs_all(g, A, Acnt) for g in G])
        self._first = None

    def all_candidates(self):
        return np.arange(len(self.answers))

    def best_guess(self, cand):
        k = len(cand)
        if k <= 2:
            return int(cand[0])
        if k == len(self.answers) and self._first is not None:
            return self._first
        sub = self.P[:, cand].astype(np.int32)
        sub += np.arange(len(self.guesses), dtype=np.int32)[:, None] * 243
        counts = np.bincount(sub.ravel(), minlength=len(self.guesses) * 243)
        p = counts.reshape(len(self.guesses), 243) / k
        with np.errstate(divide="ignore", invalid="ignore"):
            H = -np.nansum(np.where(p > 0, p * np.log2(p), 0), axis=1)
        H[cand] += 1.0 / k                 # tiny bonus: a guess that could win outright
        best = int(np.argmax(H))
        if k == len(self.answers):
            self._first = best
        return best

    def narrow(self, cand, guess_i, pat):
        return cand[self.P[guess_i, cand] == pat]

    def solve(self, answer_i, verbose=False):
        cand, trace = self.all_candidates(), []
        for turn in range(1, 20):
            g = self.best_guess(cand)
            pat = int(self.P[g, answer_i])
            trace.append(show(self.guesses[g], pat))
            if g == answer_i:
                break
            cand = self.narrow(cand, g, pat)
        if verbose:
            print("\n".join(trace))
        return turn


def cmd_play(bot):
    cand = bot.all_candidates()
    for turn in range(1, 7):
        g = bot.best_guess(cand)
        print(f"\nGuess {turn}: {bot.guesses[g].upper()}   ({len(cand)} candidates left)")
        while True:
            try:
                pat = parse_feedback(input("feedback (g/y/b): "))
                break
            except ValueError as e:
                print(e)
        if pat == 242:
            print("Solved!")
            return
        cand = bot.narrow(cand, g, pat)
        if len(cand) == 0:
            print("No word matches that feedback; check for a typo or a word missing from the list.")
            return
        if len(cand) <= 8:
            print("Remaining:", ", ".join(bot.answers[i] for i in cand))
    print("Out of guesses.")


def cmd_sim(bot, n):
    idx = range(len(bot.answers)) if n is None else np.random.default_rng(0).choice(
        len(bot.answers), n, replace=False)
    turns = [bot.solve(int(i)) for i in idx]
    dist = Counter(turns)
    print(f"first guess: {bot.guesses[bot._first].upper()}")
    for t in sorted(dist):
        print(f"{t} guesses: {dist[t]:5d}  {dist[t] / len(turns):6.1%}")
    print(f"average {np.mean(turns):.3f}   worst {max(turns)}   "
          f"failed (>6): {sum(t > 6 for t in turns)}")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("mode", choices=["play", "solve", "sim"])
    ap.add_argument("word", nargs="?")
    ap.add_argument("-n", type=int, help="sim: only a random sample of n answers")
    args = ap.parse_args()

    bot = Bot(read_words(HERE / "answers.csv"), read_words(HERE / "allowed-guesses.csv"))
    if args.mode == "play":
        cmd_play(bot)
    elif args.mode == "solve":
        bot.solve(bot.answers.index(args.word.lower()), verbose=True)
    else:
        cmd_sim(bot, args.n)
