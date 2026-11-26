"""Write labelled training questions for a trained lens, disjoint from our dev set.

    .venv/bin/python research/make_train_questions.py [--out research/data/train]

Same families and format as research/make_questions.py (chat-wrapped as in the practice set, plus
plain prompts), but other athletes, musicians and landmarks, other arithmetic problems, and other
phrasings for the calendar questions (there are only 7 days and 12 months, so the facts repeat; the
prompts never do). No prompt here appears in research/data/dev. Each question carries the token
the lens should surface (``intermediate``): the answer for single-hop questions, the step in
between for two-hop ones.
"""

import argparse
import json
import random
from pathlib import Path

from transformers import AutoTokenizer

from make_questions import CHAT, DAYS, MONTHS, MODEL_ID, SUFFIX, single_token

ATHLETES = {
    "Mario Lemieux": "hockey", "Bobby Orr": "hockey", "Gordie Howe": "hockey", "Jaromir Jagr": "hockey", "Auston Matthews": "hockey",
    "Arnold Palmer": "golf", "Annika Sorenstam": "golf", "Jordan Spieth": "golf", "Scottie Scheffler": "golf", "Gary Player": "golf",
    "Andre Agassi": "tennis", "Pete Sampras": "tennis", "Steffi Graf": "tennis", "Martina Navratilova": "tennis", "Carlos Alcaraz": "tennis", "Bjorn Borg": "tennis",
    "Magic Johnson": "basketball", "Larry Bird": "basketball", "Kevin Durant": "basketball", "Tim Duncan": "basketball", "Dirk Nowitzki": "basketball", "Giannis Antetokounmpo": "basketball",
    "Mickey Mantle": "baseball", "Ken Griffey Jr.": "baseball", "Barry Bonds": "baseball", "Mike Trout": "baseball", "Jackie Robinson": "baseball",
    "Joe Louis": "boxing", "Sugar Ray Leonard": "boxing", "Evander Holyfield": "boxing", "Lennox Lewis": "boxing", "Canelo Alvarez": "boxing",
    "Brian Lara": "cricket", "Shane Warne": "cricket", "Ricky Ponting": "cricket", "Jacques Kallis": "cricket", "MS Dhoni": "cricket",
    "Ian Thorpe": "swimming", "Ryan Lochte": "swimming", "Missy Franklin": "swimming",
    "Anatoly Karpov": "chess", "Hikaru Nakamura": "chess", "Judit Polgar": "chess", "Viswanathan Anand": "chess",
    "Miguel Indurain": "cycling", "Bernard Hinault": "cycling", "Tadej Pogacar": "cycling",
    "Jonah Lomu": "rugby", "Richie McCaw": "rugby", "Jonny Wilkinson": "rugby",
    "Hakuho": "sumo", "Chiyonofuji": "sumo",
}
MUSICIANS = {
    "Carlos Santana": "guitar", "Stevie Ray Vaughan": "guitar", "Jimmy Page": "guitar", "Eddie Van Halen": "guitar", "Slash": "guitar", "Andres Segovia": "guitar",
    "Mstislav Rostropovich": "cello", "Steven Isserlis": "cello",
    "Hilary Hahn": "violin", "Jascha Heifetz": "violin", "Anne-Sophie Mutter": "violin", "Yehudi Menuhin": "violin",
    "Chet Baker": "trumpet", "Wynton Marsalis": "trumpet", "Clifford Brown": "trumpet",
    "Sonny Rollins": "saxophone", "Stan Getz": "saxophone", "Coleman Hawkins": "saxophone",
    "Arthur Rubinstein": "piano", "Martha Argerich": "piano", "Thelonious Monk": "piano", "Vladimir Ashkenazy": "piano", "Herbie Hancock": "piano",
    "Buddy Rich": "drums", "Keith Moon": "drums", "Travis Barker": "drums", "Gene Krupa": "drums",
    "Ralph Stanley": "banjo", "Pete Seeger": "banjo",
    "Artie Shaw": "clarinet", "Jean-Pierre Rampal": "flute", "Anoushka Shankar": "sitar",
    "Jake Shimabukuro": "ukulele", "Andreas Vollenweider": "harp",
}
# landmark: (country, capital or None, language or None)
LANDMARKS = {
    "the Arc de Triomphe": ("France", "Paris", "French"), "the Palace of Versailles": ("France", "Paris", "French"), "Notre-Dame Cathedral": ("France", "Paris", "French"),
    "the Pantheon in Rome": ("Italy", "Rome", "Italian"), "St. Mark's Basilica": ("Italy", "Rome", "Italian"), "the Duomo of Florence": ("Italy", "Rome", "Italian"),
    "Park Guell": ("Spain", "Madrid", "Spanish"), "the Prado Museum": ("Spain", "Madrid", "Spanish"),
    "Meteora": ("Greece", "Athens", "Greek"), "the Temple of Poseidon at Sounion": ("Greece", "Athens", "Greek"),
    "the Red Fort": ("India", None, "Hindi"), "the Gateway of India": ("India", None, "Hindi"),
    "the Terracotta Army": ("China", "Beijing", "Chinese"), "the Temple of Heaven": ("China", "Beijing", "Chinese"),
    "Lake Titicaca's Uros islands": ("Peru", "Lima", "Spanish"),
    "Sugarloaf Mountain": ("Brazil", None, "Portuguese"),
    "the Great Barrier Reef": ("Australia", "Canberra", "English"),
    "Wadi Rum": ("Jordan", "Amman", "Arabic"),
    "Cologne Cathedral": ("Germany", "Berlin", "German"), "the Reichstag building": ("Germany", "Berlin", "German"),
    "Saint Basil's Cathedral": ("Russia", "Moscow", "Russian"), "the Hermitage Museum": ("Russia", "Moscow", "Russian"),
    "Himeji Castle": ("Japan", "Tokyo", "Japanese"), "the Fushimi Inari shrine": ("Japan", "Tokyo", "Japanese"),
    "the Valley of the Kings": ("Egypt", "Cairo", "Arabic"), "Abu Simbel": ("Egypt", "Cairo", "Arabic"),
    "Teotihuacan": ("Mexico", None, "Spanish"),
    "Cappadocia": ("Turkey", "Ankara", "Turkish"), "Ephesus": ("Turkey", "Ankara", "Turkish"),
    "Niagara Falls on the Ontario side": ("Canada", "Ottawa", None),
    "Malbork Castle": ("Poland", "Warsaw", "Polish"),
    "the Grand Place": ("Belgium", "Brussels", None),
    "the Anne Frank House": ("Netherlands", "Amsterdam", "Dutch"),
    "Tivoli Gardens": ("Denmark", "Copenhagen", "Danish"),
    "the Blue Lagoon": ("Iceland", "Reykjavik", "Icelandic"),
    "Hallstatt": ("Austria", "Vienna", "German"),
    "the Atacama Desert": ("Chile", "Santiago", "Spanish"),
    "Hoi An's Ancient Town": ("Vietnam", "Hanoi", "Vietnamese"),
    "the Vasa Museum": ("Sweden", "Stockholm", "Swedish"),
    "the Geirangerfjord": ("Norway", "Oslo", "Norwegian"),
    "the Belem Tower": ("Portugal", "Lisbon", "Portuguese"),
    "the Perito Moreno Glacier": ("Argentina", None, "Spanish"),
    "the Grand Palace in Bangkok": ("Thailand", "Bangkok", "Thai"),
    "the Maasai Mara": ("Kenya", "Nairobi", None),
}
SPORT_POOL = ["tennis", "hockey", "golf", "basketball", "baseball", "boxing", "cricket", "swimming", "chess", "cycling", "rugby", "sumo", "karate", "volleyball", "fencing", "rowing"]
INSTRUMENT_POOL = ["guitar", "cello", "violin", "trumpet", "saxophone", "piano", "drums", "banjo", "clarinet", "sitar", "flute", "harp", "accordion", "trombone", "organ", "ukulele", "oboe", "tuba"]

SPORT_TEMPLATES = ["What sport does the athlete {} play?", "Which sport is {} famous for?", "{} is a star in which sport?", "In what sport did {} compete?"]
INSTRUMENT_TEMPLATES = ["Which instrument is the musician {} known for playing?", "What instrument does {} play?", "{} is famous for playing which instrument?"]
COUNTRY_TEMPLATES = ["In which country is {}?", "Which country is {} located in?", "{} is in which country?"]
HOP_TEMPLATES = {
    "capital": ["What is the capital of the country where {} is?", "Name the capital city of the country that has {}."],
    "language": ["What language is spoken in the country where {} is?", "Which language do people speak in the country that has {}?"],
}
SHIFT_TEMPLATES = ["Which {unit} comes {word} {name}?", "What {unit} is {word} {name}?", "Name the {unit} {word} {name}.", "If it is {name}, what {unit} is it one {unit} {word}?"]
SHIFT2_TEMPLATES = ["Which {unit} comes two {unit}s {word} {name}?", "Name the {unit} that is two {unit}s {word} {name}.", "If it is {name}, what {unit} is it two {unit}s {word}?"]
ARITH_TEMPLATES = ["What is {a} {op} {b}?", "Compute {a} {op} {b}.", "How much is {a} {op} {b}?"]
ARITH2_TEMPLATES = ["What is ({a} + {b}) × {c}?", "Compute ({a} + {b}) × {c}.", "What is {c} × ({a} + {b})?"]


def options_for(rng, answer, pool, extra=()):
    distractors = [e for e in dict.fromkeys(extra) if e != answer][:4]
    rest = [p for p in pool if p != answer and p not in distractors]
    distractors += rng.sample(rest, 4 - len(distractors))
    options = [answer] + distractors
    rng.shuffle(options)
    return options


def build(rng, ok):
    countries = sorted({c for c, _, _ in LANDMARKS.values() if ok(c)})
    sports = [s for s in SPORT_POOL if ok(s)]
    instruments = [i for i in INSTRUMENT_POOL if ok(i)]
    rows = []
    for name, sport in ATHLETES.items():
        if ok(sport):
            for t in SPORT_TEMPLATES:
                rows.append(("sport", t.format(name), sport, sport, options_for(rng, sport, sports)))
    for name, instrument in MUSICIANS.items():
        if ok(instrument):
            for t in INSTRUMENT_TEMPLATES:
                rows.append(("instrument", t.format(name), instrument, instrument, options_for(rng, instrument, instruments)))
    for landmark, (country, capital, language) in LANDMARKS.items():
        if not ok(country):
            continue
        for t in COUNTRY_TEMPLATES:
            text = t.format(landmark)
            rows.append(("country", text[0].upper() + text[1:], country, country, options_for(rng, country, countries)))
        for kind, value in (("capital", capital), ("language", language)):
            if value:
                for t in HOP_TEMPLATES[kind]:
                    rows.append(("hop_country", t.format(landmark), country, value, options_for(rng, country, countries)))
    for names, unit in ((DAYS, "day"), (MONTHS, "month")):
        for i, name in enumerate(names):
            for word, step in (("after", 1), ("before", -1)):
                target = names[(i + step) % len(names)]
                for t in SHIFT_TEMPLATES:
                    rows.append(("shift1", t.format(unit=unit, word=word, name=name), target, target, options_for(rng, target, names, extra=[name])))
                middle, answer = names[(i + step) % len(names)], names[(i + 2 * step) % len(names)]
                for t in SHIFT2_TEMPLATES:
                    rows.append(("shift2", t.format(unit=unit, word=word, name=name), middle, answer, options_for(rng, middle, names, extra=[answer, name])))
    for a in range(0, 13):
        for b in range(0, 13):
            for op in "+-×":
                if op == "-" and b > a:
                    continue
                value = a + b if op == "+" else a - b if op == "-" else a * b
                if value <= 20:
                    t = rng.choice(ARITH_TEMPLATES)
                    rows.append(("arith1", t.format(a=a, op=op, b=b), str(value), str(value), options_for(rng, str(value), [str(n) for n in range(0, 21)])))
    for a in range(1, 7):
        for b in range(1, 7):
            for c in range(2, 6):
                t = rng.choice(ARITH2_TEMPLATES)
                rows.append(("arith2", t.format(a=a, b=b, c=c), str(a + b), str((a + b) * c), options_for(rng, str(a + b), [str(n) for n in range(0, 13)], extra=[str((a + b) * c), str(c)])))
    return rows


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    here = Path(__file__).parent
    parser.add_argument("--out", type=Path, default=here / "data" / "train")
    parser.add_argument("--dev", type=Path, default=here / "data" / "dev" / "questions.jsonl")
    parser.add_argument("--seed", type=int, default=1)
    parser.add_argument("--plain-share", type=float, default=0.25, help="share of questions also written as plain prompts")
    args = parser.parse_args()
    tokenizer = AutoTokenizer.from_pretrained(MODEL_ID)
    rng = random.Random(args.seed)
    dev_prompts = {json.loads(l)["prompt"] for l in open(args.dev)}

    questions, clashes = [], 0
    for n, (family, prompt, intermediate, answer, options) in enumerate(build(rng, lambda w: single_token(tokenizer, w))):
        prompt = prompt + SUFFIX
        if prompt in dev_prompts:
            clashes += 1
            continue
        styles = [("chat", CHAT.format(prompt))] + ([("plain", prompt)] if rng.random() < args.plain_share else [])
        for style, model_input in styles:
            ids = tokenizer(model_input, add_special_tokens=False)["input_ids"]
            questions.append({"id": f"train-{family}-{n:04d}" + ("-plain" if style == "plain" else ""), "family": family, "style": style,
                              "prompt": prompt, "model_input": model_input, "read_position": len(ids) - 1,
                              "read_token": tokenizer.decode([ids[-1]]), "intermediate": intermediate, "answer": answer, "options": options})
    args.out.mkdir(parents=True, exist_ok=True)
    with open(args.out / "questions.jsonl", "w") as f:
        for q in questions:
            f.write(json.dumps(q, ensure_ascii=False) + "\n")
    counts = {}
    for q in questions:
        counts[q["family"]] = counts.get(q["family"], 0) + 1
    print(f"wrote {len(questions)} questions ({clashes} dropped as dev prompts): {counts}")


if __name__ == "__main__":
    main()
