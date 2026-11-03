# Annotation guide / Transkripsiya qoidalari

How to correct transcripts in `review_app.py`. Read it once; keep it open for the first day.

## Asosiy qoidalar (key rules in Uzbek)

1. **Eshitilganini yozing.** Gapiruvchining grammatikasini to'g'rilamang, so'z qo'shmang, so'z tashlab ketmang.
2. **Faqat lotin yozuvi.** Kirill harflarini ishlatmang.
3. **Apostrof: faqat oddiy `'`** — o', g', va tutuq belgisi uchun ham: *o'zbek, g'oya, ma'lumot, san'at*.
4. **Raqamlarni so'z bilan yozing:** *yuz ming so'm*, *to'qson to'qqiz*, *ikki ming yigirma beshinchi yil*. Telefon raqamlari ham so'z bilan.
5. **Ruscha so'zlar — eshitilganidek, lotincha:** *konechno, seychas, zayavka, karta*.
6. **Tushunarsiz so'z → `[?]`.** Taxmin qilmang.
7. **"Ee", "mm" kabi tovushlarni yozmang.** Lekin *ha, xo'p, mayli, endi, ana* — bular so'z, yozing.
8. **Yomon audio** tugmasi: faqat shovqin/musiqa/jimlik, yoki gapning yarmidan ko'pi tushunarsiz.

## Rules in detail (English)

**Script and spelling**
- Latin script only. Use standard Uzbek spelling for the word the speaker said, even if their pronunciation is dialectal. Don't swap it for a different word, and don't fix their grammar. Contractions stay as spoken: *qoldirgandim*, not *qoldirgan edim*.
- Use the plain apostrophe `'` for every apostrophe: *o'*, *g'*, and the tutuq belgisi (*ma'lumot*). The training code normalises the other forms (ʻ ʼ ‘ ’) anyway, but typing one form keeps things simple.
- Punctuation: a capital letter at the start of a sentence, and `.` `,` `?` where they're obvious. Don't spend time on commas.

**Numbers and names**
- Numbers in words, including amounts, dates, times, phone and card numbers: *sakkiz yuz o'ttiz besh*. A digit-by-digit reading stays digit by digit: *nol to'qson*.
- People's names, streets, cities: standard Latin spelling (*Dilnoza, Chilonzor*).
- Brand and product names: their usual written form (*Click, Payme, Telegram*).

**Russian and mixed speech**
- Russian words inside Uzbek speech: write them as heard, in Latin, using Uzbek letters (*sh, ch, ya, yo, ts*): *pojaluysta, nomer zakaza, dogovor*.
- **A whole clip in Russian:** press **Bad audio** for now. *(To be confirmed: we may want to keep these later.)*

**Hesitations, fillers, cut-off words**
- Don't write pure sounds: *ee, aa, mm, hmm*.
- Write real words even when they're used as fillers: *ha, yo'q, xo'p, mayli, endi, ana, haligi*.
- Repeated words that were actually said: write them (*men men bordim*).
- A word cut off halfway: if you can tell which word it is, write the whole word; otherwise leave it out.

**Two voices, overlap**
- Stereo clips contain one side of the call. Transcribe the main speaker. Faint bleed from the other side: ignore it.
- Mono clips: write both speakers in the order they speak, with no speaker labels.
- Overlapping speech you can't separate: `[?]`.

**Unclear audio**
- One word or a few you can't make out: `[?]` for each unclear stretch. Never guess a plausible word. A wrong word is worse than `[?]`.
- **Bad audio** when the clip has only noise, music, a hold tone, beeps or silence, or when more than half of it is unintelligible.
- Automated voice messages (IVR, "qo'ng'irog'ingiz yozib olinmoqda"): transcribe them normally. They're valid speech.

**The model's draft**
- The draft is often wrong in a way that sounds plausible. **Listen to the whole clip** before saving, even when the text looks right.
- Delete text the model invented: repeated sentences, *"Subtitrlar..."*, *"Obuna bo'ling"*, anything not in the audio.
- If the draft is correct, just press Save. Correct clips are valuable too.

## Examples

| Model draft | Corrected | Why |
|---|---|---|
| Asalom alaykum operator Dilnoza sizga qanday yordam bera olaman | Assalomu alaykum, operator Dilnoza, sizga qanday yordam bera olaman? | Spelling + basic punctuation |
| 150000 so'm to'ladim | Yuz ellik ming so'm to'ladim. | Numbers as words, exactly as the speaker said them |
| Заявка qoldirgandim | Zayavka qoldirgandim. | Latin only; Russian word as heard; the contraction stays |
| Ee mm buyurtmam hali kelmadi | Buyurtmam hali kelmadi. | Pure filler sounds removed |
| Men Yunusobod filialiga bordim | Men [?] filialiga bordim. | Speaker said an unclear place name; the model guessed |
| Rahmat. Rahmat. Rahmat. Rahmat. | Rahmat. | Speaker said it once; the model repeated it |
| Subtitrlar Amara.org tomonidan | *(Bad audio)* | Hold music; the model invented the text |

## How the training code uses your labels
- **Bad audio** clips are never used for training.
- Clips containing `[?]` are excluded from training and from test scoring.
