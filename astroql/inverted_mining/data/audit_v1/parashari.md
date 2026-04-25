# Inverted-mining audit: **parashari** longevity (father)

_43 unique passages retrieved across all subjects. Showing top 32 clusters with frequency ≥ 2._

Each cluster = one classical-text chunk that the 8-pass RAG retrieved for one or more death-state snapshots. Higher frequency = more snapshots' RAG queries matched this chunk = higher chance the chunk encodes a generalizable rule.

**Review prompt for each cluster**: is this passage actually describing a *rule* for father-death timing (vs. general definition / prose / off-topic)? If yes, what AstroQL feature paths would encode it?

---

## C1. freq=97 — `ia_Neelakanta_Ashtakavarga.txt`
**Feature hints**: yogakaraka
**Gender split**: M=48 F=49
**Age-bucket split**: 15-30=22, 30-45=20, 45-60=22, 60+=12, <15=21

```
--- Page 1 ---
ASHTAKAVARGA SYSTEM OF PREDICTION – PART 1. 
 
 
We know that in order to come to certain definite conclusions, we need to assess not only 
the position of planets in the natal chart but also in transit.  The transit of planets can 
either support or oppose the natal chart.  There are some learned scholars who opine that 
when transits of planets oppose the planets in natal chart, the gochara scheme of things 
gains the upper hand.  Now a days lots of people are anxious to know when Jupiter or 
Saturn will leave a particular sign and enter the next.  One of the most dreaded transits if 
the sade-sathi of Saturn.  But do all people perceive effect of gochara in equal measure?  
The answer is “NO”.  So a convenient explanation is given saying that Saturn in his 
second round is beneficial, third round is not so, etc.  But the real clue to deciphering the 
effects of planets in transit, lies in our comprehending the Ashtaka Varga system of 
Prediction.  Now we will take a brief look at the various ways in which Astrology is used 
to delineate the prospects for future. 
 
a)  Natal Astrology: 
 
 
For an individual the birth chart is the main building block from which all other 
results flow.  Some times a query is raised about the results of Jupiter or Saturn in transit. 
If in the birth chart, Saturn is either the lagna lord or a yogakaraka or is well placed in 
faourable houses, then one need not worry at all where Saturn is placed in tran […truncated…]
```
**Sample subjects** (top 5 by score): Olof Palme (M/<15, score=0.271); George W. Bush (M/60+, score=0.271); James Dean (M/60+, score=0.271); Liza Minnelli (F/30-45, score=0.271); Drew Barrymore (F/15-30, score=0.271)

**Reviewer notes**:
- Relevance: ☐ rule  ☐ definition  ☐ prose  ☐ off-topic
- Feature path(s):
- Yoga draft:

---

## C2. freq=97 — `transit_weighting_classical_methodology.txt`
**Feature hints**: dusthana, neecha_bhanga
**Gender split**: M=48 F=49
**Age-bucket split**: 15-30=22, 30-45=20, 45-60=22, 60+=12, <15=21

```
PENALTY 2: DEBILITATED SIGNIFICATOR WITHOUT BHANGA
- Significator in debilitation sign and no Neechabhanga:
- DEMOTION: −2 confidence points.
- Apply BHANGA TEST first: if dispositor in kendra from Lagna OR Moon,
  if exaltation lord in kendra, if exchange of signs, debilitation is
  cancelled — do NOT apply this penalty.


PENALTY 3: DUSTHANA LORD PD WITHOUT BHANGA
- 6th, 8th, or 12th lord running as Pratyantardasha without supporting
  benefic dignity:
- DEMOTION: −1 confidence point for events related to gain/manifestation.
- For events related to loss/transformation/spirituality (8H matters,
  12H matters), this is NOT a penalty — it's appropriate.


PENALTY 4: ECLIPSE WITHIN 7 DAYS OF TRIGGER DATE
- Solar or lunar eclipse within 7 days before or after the trigger date:
- DEMOTION: −1 confidence point AND possible event reschedule.
- Major events scheduled in eclipse windows often get postponed by 1-2
  months even when astrological factors are otherwise favorable.


PENALTY 5: TRIGGER FALLS IN DASHA SANDHI (JUNCTION)
- Event date falls in the last 5% of an old dasha and first 5% of a new
  dasha (Sandhi):
- DEMOTION: −1 confidence point.
- Sandhi is classically described as "sthana balahina" — positionally weak.


==============================================================
SECTION D: TRADITION-SPECIFIC EVENT-TYPE WEIGHTING
==============================================================


KP (Krishnamurti Padhdhati):
- Best for: IMMEDIATE events within 6-24 months, prec […truncated…]
```
**Sample subjects** (top 5 by score): Olof Palme (M/<15, score=0.299); George W. Bush (M/60+, score=0.299); James Dean (M/60+, score=0.299); Liza Minnelli (F/30-45, score=0.299); Drew Barrymore (F/15-30, score=0.299)

**Reviewer notes**:
- Relevance: ☐ rule  ☐ definition  ☐ prose  ☐ off-topic
- Feature path(s):
- Yoga draft:

---

## C3. freq=97 — `ia_Analysing Horoscope through modern techniques_Mehta_djvu.txt`
**Feature hints**: h_9
**Gender split**: M=48 F=49
**Age-bucket split**: 15-30=22, 30-45=20, 45-60=22, 60+=12, <15=21

```
VIN— Transit in Kakshaya 
This is an important concept in Ashtak Varga. Each Rasi of 30 
degrees is divided into eight equal parts of 3° 45'. The lords of 
these kakshaya are Saturn, Jupiter, Mars, Sun, Venus, Mercury, 
Moon and Lagna in that order. A planet while transiting in a kakshaya 
whose lord has contributed a Bindu (this can be ascertained from 
Prasthar Chakra of each planet) gives good results, while in 
transiting in a Binduless kakshaya evil results are manifested. 
There is another concept of interpretation of results. 
Suppose Saturn in transit reaches the kakshaya of Mars, and Mars . 
has contributed one Bindu, then the result of transit in this kakshaya 
will not only be good, but Satur here wilt give the results of its own 
signification as also that of Mars. This has been illustrated with 


300 Analysing Horoscope Through Modern Techniques 


examples in the chapter on kakshaya in my other book. 
IX — Temporal Benefic and Malefic for different Lagnas 
While judging the result of transit of a planet its lordship must be 
taken into account whether it is benefic or malefic for a particular 
Lagna. This concept is peculiar to Vedic Astrology and has been 
elaborated by Maharishi Parasara in Brihat Parasara Hora Shastra. 
In this connection following principles should be memorized; 
i) Lagna Lord is always Benefic ie. Mars for Aries and 
Scorpio Lagnas and Satum for Cancer and Aquarius are 
benefics. 
ii) Lord of 5th house and lords of 9th house are always 
au […truncated…]
```
**Sample subjects** (top 5 by score): Olof Palme (M/<15, score=0.245); George W. Bush (M/60+, score=0.245); James Dean (M/60+, score=0.245); Liza Minnelli (F/30-45, score=0.245); Drew Barrymore (F/15-30, score=0.245)

**Reviewer notes**:
- Relevance: ☐ rule  ☐ definition  ☐ prose  ☐ off-topic
- Feature path(s):
- Yoga draft:

---

## C4. freq=97 — `chart_dignity_exceptions_methodology.txt`
**Feature hints**: —
**Gender split**: M=48 F=49
**Age-bucket split**: 15-30=22, 30-45=20, 45-60=22, 60+=12, <15=21

```
RULE 2: VARGOTTAMA — SAME SIGN IN D1 AND D9

A planet occupying the SAME rashi in both the Rashi chart (D1) and the
Navamsha chart (D9) is called Vargottama. Vargottama amplifies the planet's
intrinsic dignity by a multiple — a Vargottama planet acts as if it were
twice as strong in any timing role.

EXCEPTION RULE: A Vargottama planet's dasha gives stable, sustained results
in the area it signifies. Vargottama Lagna lord = consistent vitality and
self-direction. Vargottama 7th lord = stable marriage. Vargottama 10th lord
= durable career. When timing an event, prefer the dasha of a Vargottama
planet over a non-Vargottama planet of equal house lordship.

Reference: Brihat Parashar Hora Shastra (Navamsha Adhyaya), Jataka Parijata,
Saravali. Specifically, see CS Patel "Navamsa in Astrology" for detailed
case studies of Vargottama timing.
```
**Sample subjects** (top 5 by score): Olof Palme (M/<15, score=0.229); George W. Bush (M/60+, score=0.229); James Dean (M/60+, score=0.229); Liza Minnelli (F/30-45, score=0.229); Drew Barrymore (F/15-30, score=0.229)

**Reviewer notes**:
- Relevance: ☐ rule  ☐ definition  ☐ prose  ☐ off-topic
- Feature path(s):
- Yoga draft:

---

## C5. freq=97 — `transit_weighting_classical_methodology.txt`
**Feature hints**: maraka, badhaka, dusthana, neecha_bhanga, yogakaraka
**Gender split**: M=48 F=49
**Age-bucket split**: 15-30=22, 30-45=20, 45-60=22, 60+=12, <15=21

```
TRANSIT WEIGHTING — CLASSICAL VEDIC METHODOLOGY FOR RANKING COMPETING WINDOWS

This document codifies the classical rules for weighing transits (gochar) and
ranking competing dasha-transit windows. These rules apply UNIVERSALLY across
all life events: marriage, childbirth, career, finance, education, health,
death (maraka), litigation. The same weighting hierarchy governs every
prediction — only the target houses and karakas change by topic.

VOCABULARY USED IN CLASSICAL TEXTS:
gochar (transit), drishti (aspect), kendra (1/4/7/10), trikona (1/5/9),
dusthana (6/8/12), upachaya (3/6/10/11), maraka (2/7), badhaka, sandhi
(junction), kakshya, bindus (Ashtakavarga points), bhukti, antaradasha,
pratyantardasha, sookshma, uchcha (exalted), neecha (debilitated),
swakshetra (own sign), mooltrikona, vargottama, neechabhanga (cancellation
of debilitation), shubha (benefic), paap (malefic), balavan (strong),
durbal (weak), paaka (dasha lord), bhoga (transit influence), Yogakaraka,
Atmakaraka, Darakaraka, Putrakaraka, Bhratrukaraka, Chara Karaka, sthira
karaka, Phala (fruit/result).


==============================================================
SECTION A: STANDARD WEIGHTING RULES
==============================================================
```
**Sample subjects** (top 5 by score): Olof Palme (M/<15, score=0.180); George W. Bush (M/60+, score=0.180); James Dean (M/60+, score=0.180); Liza Minnelli (F/30-45, score=0.180); Drew Barrymore (F/15-30, score=0.180)

**Reviewer notes**:
- Relevance: ☐ rule  ☐ definition  ☐ prose  ☐ off-topic
- Feature path(s):
- Yoga draft:

---

## C6. freq=48 — `P.V.R._Narasimha_Rao_-_Vedic_Astrology_Textbook.pdf`
**Feature hints**: h_9, father_specific, mother_specific
**Gender split**: M=17 F=31
**Age-bucket split**: 15-30=11, 30-45=16, 45-60=8, 60+=5, <15=8

```
It shows misfortune, suffering and death. 
Using it, death of near relatives can also be timed. 
For each relative, we treat the house that shows him/her as lagna and find Shoola 
dasa based on it. Shoola dasa starts from the stronger of that rasi and the 7th from it. 
We have different names for Shoola dasas started from different houses. For 
example, Pitri Shoola dasa (pitri = father) starts from the stronger of 9th and 3rd
houses. It is used in the timing of father’s death. Bhratri Shoola dasa (bhratri = 
brother) also starts from the stronger of 3rd and 9th houses and it shows the death of 
younger siblings. Matri Shoola dasa (matri = mother) starts from the stronger of 4th
and 10th houses. Dara Shoola dasa (dara = wife) starts from the stronger of 7th and 1st
houses and it shows the death of wife (this dasa will be identical to the native’s 
normal Shoola dasa). Putra Shoola dasa (putra = son) starts from the stronger of 5th
and 11th houses and it shows the death of a child. 
We mentioned earlier that sthira karakas are useful in timing death. Sthira karaka of 
father represents father in a chart, for the purpose of matters controlled by Shiva (i.e.
suffering and death). Pitri Shoola dasa shows the motion of Shiva’s force for the 
father. When Shiva’s force strikes trines from sthira karaka of father, father’s death 
can take place. Trines from the corresponding arudha pada can also give death. 
Similarly, we can time the death of other near relatives. […truncated…]
```
**Sample subjects** (top 5 by score): Barack Obama (M/15-30, score=0.227); Alice Munro (F/30-45, score=0.242); Michael Douglas (M/60+, score=0.244); Roger Boutet de Monvel (M/30-45, score=0.246); Theo Mann-Bouwmeester (F/15-30, score=0.247)

**Reviewer notes**:
- Relevance: ☐ rule  ☐ definition  ☐ prose  ☐ off-topic
- Feature path(s):
- Yoga draft:

---

## C7. freq=45 — `BPHS - 2 RSanthanam.pdf`
**Feature hints**: father_specific
**Gender split**: M=30 F=15
**Age-bucket split**: 15-30=10, 30-45=5, 45-60=12, 60+=8, <15=10

```
il;;;ht; tn" sutoto will pass in transit
through Uttarabhadtl *i" Ititona nakshatras Pushyami ct
Anuradha, thc deathliin" f"ntt will take place if an inaus�picious Dasa be it t;;;; "itnut 
tima Should the Dasa be
auspicious o, f"no.uro'bi, the father will be in great distress'
gqtf q'tqfqq rt aril
a.tq5n{nqfr qqr rrtsFd
aqktlorqi6
ftweernqt
crfr fr"m*
I G' I I . If the Ashtakavarga rekha lumber is urultiplied b1
tn" voga pinda and;;;;ffi is divided bv f2'.the'r"t-iT5]
will denote the rasr through which-or through the rasts ri
trikona to it, the ,run'i'oi 5u'urn will cause harm or unfavou
rable efrects to father". ;;'h of tbe father uiay occur if th
Dasa prevailing at tnai time be unfavourable' If the Dasa b
favourable father *itt foo only adverse effects (like seriou
illness).
Illustration-The Sun is in Capricorn' The nineth lT1-f'tot
- it is Virgo. Multiply its Ashatakavarga number by 148 (YoS
Pinda), Divide tt'"--p'oJ""t 296 by 12' The remainder
teprcsents Scorpio inJ'*[""" Rasis of which arc Canct
rn anfnqmiqq t
qr{fl: llto11
ain a?q t
d s$i qtsq_'lrqrilqlll!E4 Erihat thrasara Hora &sta
and Pisces. When Saturn transits through any of these rasis, death or distress to father may be predic"ted.
Aristr to father
emfqq gfil {Af q?a n qmr
Wqt ;Rs ttlRtt trcq*qTuq* frqAr
iriq q€r{ T€Tqfi nt qrig*slu r
qrqg.e gil lrfq frEtrri rt! g: u11rl qrilE grrtaf<na{{Trfre@: r
12-14. The death.of the father may be cxpect:d if Rahu, Saturn or Mars are in the Ct […truncated…]
```
**Sample subjects** (top 5 by score): Abigail Kapiolani Kawānanakoa (F/<15, score=0.246); Steve McQueen (M/15-30, score=0.246); Melanie Griffith (F/30-45, score=0.247); Adolphe, Grand Duke of Luxembourg (M/15-30, score=0.248); Al Gore (M/45-60, score=0.249)

**Reviewer notes**:
- Relevance: ☐ rule  ☐ definition  ☐ prose  ☐ off-topic
- Feature path(s):
- Yoga draft:

---

## C8. freq=45 — `ia_Ashtakavarga_C.S. Patel_1996 ed_djvu.txt`
**Feature hints**: h_9, father_specific, mother_specific
**Gender split**: M=19 F=26
**Age-bucket split**: 15-30=10, 30-45=10, 45-60=10, 60+=7, <15=8

```
meratrecargant akat Veritset | 

APT SATS: AH eat ATTA USAT TAT: 118 1 
Sloka 9 — When the Sun, in conjunction with the Moon, Saturn 
and Mercury, is in a Kendra at birth and that bhava has 2 net 
bindus, i.e., after both the reductions, persons learned in Astrol- 


ogy say that the father of the native will have immense admin- 
istrative power and fame ten years after the native's birth. 


Sloka 10 — A person born having Rahu, Satum or Mars in the 
4th house from the Sun, will be the cause of his father's early 
death provided that house (4th house from the Sun) is not 
aspected either by Jupiter or Venus. 


TAT ART GOTTA OTe TTT Ae 

Paar Tear Ha ATT aTTATT 1 1 

SMT RAHAT astra | 
Slokas 11- (1-1/2 — When Satum transits the 9th house from 
the Lagna or the Moon, the demise of one's parents may be 
predicted, if he (Saturn) is aspected by or associated with a 


malefic planet and also the Dasa and Antardasa at that time 
indicate the same. 


ASHTAKAVARGA 8] 


wrargasestteremar a freer vez 
FoMAeMA YF ACATATT AAA: 1123 11 


Slokas 12-13 — The demise of the father may oceur 
also during the Dasa of the planet who is the lord of the rasi 
occupied by the lord of the 4th house from the Lagna, or it may 
happen probably during the Dasa of the lord of the 4th house. 

Devakeralam and other works:- 

1. When the Sun is in a Kendra and even when that is 
a friendly house, associated with 3, 4 or 5 bindus, the father of 
the native will meet with death or suffer afflictions i […truncated…]
```
**Sample subjects** (top 5 by score): Andreas Aabel (M/45-60, score=0.250); Billy Joel (M/60+, score=0.252); Steve McQueen (M/15-30, score=0.253); Gianna Bryant (F/<15, score=0.253); Margarethe Cammermeyer (F/<15, score=0.255)

**Reviewer notes**:
- Relevance: ☐ rule  ☐ definition  ☐ prose  ☐ off-topic
- Feature path(s):
- Yoga draft:

---

## C9. freq=38 — `ia_Sanjay Rath - Crux of Vedic Astrology-Timing of Events (1998)_2_djvu.txt`
**Feature hints**: h_9, h_12, dasha_chain, father_specific, mother_specific
**Gender split**: M=18 F=20
**Age-bucket split**: 15-30=9, 30-45=5, 45-60=9, 60+=4, <15=11

```
Saravali 
teaches that all the Rajyogas and Mahapurush yogas 
become non-functional if the luminaries Sun & Moon 
are debilitated or weak. If the Sun is weak, the yogas 
will never start and if the Moon is weak, the yogas 
cannot be sustained. Both Mercury and Venus are 
flanked by the luminaries and it will seem, viewed 
only from the ascendant, that a period of Rajyoga is 
about to begin. 

However, from Arudha Lagna (AL in Pisces) the 
third and eighth houses are very evil and the presence 
of Mercury and Venus in the eight house in Libra will 
threaten the life of the native. Since this is the fourth 
house from Lagna ruling vehicles, the danger will 
come from vehicles. 

Mercury as the third lord shows short journey 
was being made from Lucknow while Venus will 
reinforce the vehicle accident part as it rules travels. 
The satrupada (A6) is in Aquarius and is aspected by 
Mercury and Venus. This Mercury and Venus will 
give inimical results in the chart. Being in the ninth 
house from the satrupada these planets will protect 
the enemy/evil shasta forces and will bring ruin on 
the native. Thus, the entire rosy picture of Malavya 
Mahapurush yoga etc. is completely altered when the 


238 The Crux of Vedic Astrology-Timing of Events 

hidden nature of the Mercury and Venus are revealed 
by the AL and A6. 

A rare combination of Saturn and Ketu are in the 
twelfth house from Lagna. If Saturn conjoins Rahu/ 
Ketu in the first to sixth house from Lagna, the last rites 
of  […truncated…]
```
**Sample subjects** (top 5 by score): Jim Morrison (M/60+, score=0.242); Carla Bruni (F/15-30, score=0.243); Theo Mann-Bouwmeester (F/15-30, score=0.246); Jennifer Ehle (F/45-60, score=0.246); Laurent de Brunhoff (M/<15, score=0.246)

**Reviewer notes**:
- Relevance: ☐ rule  ☐ definition  ☐ prose  ☐ off-topic
- Feature path(s):
- Yoga draft:

---

## C10. freq=28 — `ia_MN_Kedar_Predictive_Astrology.txt`
**Feature hints**: h_8, h_12, maraka, eighth_lord, ninth_lord, dasha_chain, jupiter_benefic
**Gender split**: M=12 F=16
**Age-bucket split**: 15-30=3, 30-45=5, 45-60=8, 60+=5, <15=7

```
Jupiter is also not very strong being
posited in the sign of Mercury, in sixth house
and afflicted by Saturn and Rahu. Jupiter will


--- Page 354 ---
Predictive Astrology
4.
also become Maraka, being in 12th f rom lagna
lord, in sixth from lagna, in seventh from the
Moon and is also lord of 12th house.
Karaka for First house the Sun - The Sun
is posited in enemy sign, with enemy Venus
and afflicted by Mars.
8th lord and 8th house are afflicted by Saturn.
The third house is aspected by Saturn. The
lord of third posited in friend's house, but
aspected by Saturn.
Saturn, the Karaka for longevity 
is strong being
retrograde 
and posited in friend's house. lt is
good position. With Rahu, in 6th house,
involved in exchange of 6th and 11th (both
inauspicious 
houses) and being lord of sixth
will prove helpless before Mercury (who
happens to be Maraka in this case.)
Conclusion; From the above (1 to 7) study of
the principle 11.11(1), indicates that the
longevity is going to be reduced and not
enhanced.
In third and eleventh, no malefic is posited and
in 6th house two malef ics and one benefic (not
favourable).
Benefics in kendra -The benefics in kendra
increases longevity. 
The two malef ics and one
benefic (which happens to be Maraka) are in
kendra.
8th house, if owned by a malef ic and'also
aspected by a malefic (not favourable).
Conclusion : None of the principles laid down
in para 11.11 are favourable 
for long life.
Gaj Kesari Yoga - […truncated…]
```
**Sample subjects** (top 5 by score): Fiamma Izzo (F/45-60, score=0.247); Hervé Bazin (M/30-45, score=0.249); Jennifer Coolidge (F/45-60, score=0.250); Sylvia Plath (F/<15, score=0.250); Jim Morrison (M/60+, score=0.253)

**Reviewer notes**:
- Relevance: ☐ rule  ☐ definition  ☐ prose  ☐ off-topic
- Feature path(s):
- Yoga draft:

---

## C11. freq=27 — `ia_Brihat Parāśara Horā Śhāstra By R. Santhanam_djvu.txt`
**Feature hints**: father_specific, mother_specific
**Gender split**: M=17 F=10
**Age-bucket split**: 15-30=8, 30-45=4, 45-60=7, 60+=3, <15=5

```
1-6: The matters to be considered from the Sun and other planets are as 

follows - 


The 

Sun 

The 

Moon 

Mars 

Mercur 

y 

Jupiter 

Venus 

Saturn 


The soul (3TifRT), nature, physical strength and joys and sorrows 
and father. 

Mind, wisdom, joy and mother. 

Co-born, strength, qualities and land. 

Business dealings, livelihood and friends. 

Nourishment of the body, learning, son (children), wealth and 
property. 

Marriage, enjoyments, conveyance, prostitute and sexual 
intercourse with women. 

Longevity, source of maintenance, sorrows, danger, losses and 
death. 


Page 716 



The following procedure should be adopted to ascertain the effects of a house. 
Multiply the number of rekhas with the Yoga Pinda (Rashi Pinda plus Graha Pinda) 
connected with the Ashtaka-Varga of that planet and divide the product by 27. The 
remainder will denote the number of the Nakshatra beginning from Ashwini. During 
the transit of Saturn in that Nakshatra the house (Bhava) concerned will be harmed. 
In other words, the effects of that house win become unfavourable. 

The Sun's Ashtaka-Varga 

3T5pff5r8f3rf^ift ftas? TfRPT I 

WcafEisrazn $>T3f qif§r HTJST I 
*T£TT 8f£IT ftaeSFrteft Ugaftfat g TRRf II dll 
arfel'cPtupi'al' gift Ptzn PtaRTRtsftt gi I 
IRW 2 R 2 TSTFftgroifigtgjgeter mil 

7-9: The 9th hons from the Sun at the time of birth deals with father. The, 
ekhas of the rashi (of that house) as marked in the Sun's Ashtaka-Varga should be 
multiplied by the Yoga Pinda and […truncated…]
```
**Sample subjects** (top 5 by score): Abigail Kapiolani Kawānanakoa (F/<15, score=0.254); Jimmy Carter (M/15-30, score=0.254); Adolphe, Grand Duke of Luxembourg (M/15-30, score=0.258); Steve McQueen (M/15-30, score=0.260); Rita Dalla Chiesa (F/30-45, score=0.260)

**Reviewer notes**:
- Relevance: ☐ rule  ☐ definition  ☐ prose  ☐ off-topic
- Feature path(s):
- Yoga draft:

---

## C12. freq=17 — `ia_Brihat Parāśara Horā Śhāstra By R. Santhanam_djvu.txt`
**Feature hints**: father_specific
**Gender split**: M=8 F=9
**Age-bucket split**: 15-30=6, 30-45=5, 45-60=3, <15=3

```
inauspicious Dasha be in operation at that time. Should the Dasha he auspicious or 
favourable the father will be in great distress. 


gi gsnf^eRwrai i 

3ig5Te^ang?ft3f gen ggsfSr ■ni'pi: n ?oii 

gftggjftrafe gift fterg?^ gen geta i 
fKmerer?ngi ^nwi gs^nngen n ??n 


10-11. If the Ashtaka Varga rekha number is multiplied by the Yoga Pinda 
and the product is divided by 12, the remainder will denote the rashi through which or 
through the rashis ii Trikona to it, the transit of Saturn will cause harm or unfavourable 
effects to father. Death of the father may occur if the Dasha prevailing at that time be 
unfavourable. If the Dasha b favourable father will face only adverse effects (like 
serious illness). 

Illustration-The Sun is in Capricorn. The ninth rashi from it is Virgo. Multiply its 
AshatakaVarga number by 148 (Yoga Pinda), Divide the product 296 by 12. The 
remainder represents Scorpio the Trikona Rashi of which are Cane, and Pisces. When 
Saturn transits through any of these rashis, death or distress to father may be 
predicted. 

Arista to father 


3igjf82{ sgjft ?i# jt 4 gr ijftr-tft i 

ftar?T 5Tig8) g?- n ??u 
cTJTTg gpSfTST ■rjTR-Sft gfz) ■J(?RJ2)S5gT | 
qftspfrt gz) gift PtaFiisi gsfrei fs= n ?3 n 
efgrg g^wrf^gei^irag^ ft^ara i 
3f3^g?ng>i^ %f§r 1?Pag ftg?i# ii ?«n 

12-14. The death of the father may be expected if Rahu, Saturn or Mars are 
in the 4th from the Sun at the time of transit of Saturn through any of the above three 
Rashis (Trikona Rashis). […truncated…]
```
**Sample subjects** (top 5 by score): Abigail Kapiolani Kawānanakoa (F/<15, score=0.240); Adolphe, Grand Duke of Luxembourg (M/15-30, score=0.242); Melanie Griffith (F/30-45, score=0.242); Rita Dalla Chiesa (F/30-45, score=0.243); Wilhelmina Drucker (F/30-45, score=0.245)

**Reviewer notes**:
- Relevance: ☐ rule  ☐ definition  ☐ prose  ☐ off-topic
- Feature path(s):
- Yoga draft:

---

## C13. freq=13 — `transit_weighting_classical_methodology.txt`
**Feature hints**: —
**Gender split**: M=9 F=4
**Age-bucket split**: 15-30=2, 30-45=5, 45-60=2, 60+=3, <15=1

```
==============================================================
SECTION A: STANDARD WEIGHTING RULES
==============================================================


RULE 1: ORB TIGHTNESS TIERS (GOCHAR PHALA)

The orb (angular distance) of a transiting planet from a natal point determines
the timing precision of the event:

- TIER A (orb 0.0° – 0.5°): EXACT trigger. Event manifests within days of
  the orb peak. This is the supreme classical timing window.
- TIER B (orb 0.5° – 1.0°): TIGHT trigger. Event manifests within ~2 weeks
  of the orb peak. Strongly favored.
- TIER C (orb 1.0° – 3.0°): MODERATE trigger. Event possible but requires
  confluence with dasha and other supporting factors.
- TIER D (orb > 3.0°): WIDE / supportive only. Not a primary trigger.

NUANCE: Orb tiers extend by 50% when the transiting planet is in own sign
(swakshetra), exaltation (uchcha), or moolatrikona. Per Phaladeepika,
"Balavan grahas extend their kshetra of activation."

Reference: Phaladeepika (Gochar Phala Adhyaya), Bhrigu Nadi (Gochar
Principle), BPHS (Gochar Phala Adhyaya).
```
**Sample subjects** (top 5 by score): George W. Bush (M/60+, score=0.198); James Dean (M/60+, score=0.198); Ray Bradbury (M/30-45, score=0.198); Rex Harrison (M/30-45, score=0.198); Andreas Aabel (M/45-60, score=0.198)

**Reviewer notes**:
- Relevance: ☐ rule  ☐ definition  ☐ prose  ☐ off-topic
- Feature path(s):
- Yoga draft:

---

## C14. freq=13 — `ia_MN_Kedar_Predictive_Astrology.txt`
**Feature hints**: h_7, h_8, h_12, maraka, eighth_lord, aspect_lord, father_specific, mother_specific
**Gender split**: M=3 F=10
**Age-bucket split**: 15-30=4, 30-45=1, 45-60=4, 60+=2, <15=2

```
16 
lf Yoga as suggested by the Rishis are not


--- Page 350 ---
Predictive Astrology
361
't7
18
19
20
present in the chart, the chart will be a weak
one and the native will not prosper in his life.
The future of the native depends on the yogas
or the dasha balance at birth. Elucidate.
lf lords of 4th,7th and 1Oth are posited in 6th,
8th and 12th houses (rvhich are not their own
house) the native will get prosperity- 
Comment.
What planets and houses throw light on the
nature, outlook, activities etc. of the native?
Can we predict about success and failures in
the life of the native?
Astrology is a science of sciences a wonderf u[
friend of humanity. Comment.


--- Page 351 ---
ANNEXURE I
{Refer para 11.5(ii)}
TABLE For Day Birth Only
DINA MRITU
DINA ROGA
1. 
Bharani
lV quarter
ll quarter
2.
Mrioshira
lV quarter
3. 
Ardra
ll quarter
4. 
Aslesha
lll quarter
I quarter
5. 
U. Phalquni
lll quarter
6. 
Hasta
I quarter
7. 
Swati
lV oaurter
8. 
Visakha
ll quarter
9. 
Mula
lV quarter
ll quarter
10. Sravana
lll qaurter
1 1. Dhanishtha
I qaurter
12. U. Bhadra
lll quarter
I quarter
Note : For night birth no such doshas.


--- Page 352 ---
ANNEXURE II
{Refer Para 1 1.5(iv)}
Bad Effect of Constellations
GANDA - MOOLA
Pushya
P.Asadha
Chitra
I
PORTENT FOR FATHER
il
PORTENT FOR MOTHER
ilt
HARMFUL FOR NATIVE
IV
HARMFUL FOR MATERNAL 
UNCLE


--- Page 353 ---
ANNEXURE III
{Refer Para 11.171
The Mediu […truncated…]
```
**Sample subjects** (top 5 by score): Irene Forbes-Mosse (F/15-30, score=0.252); Emmi Bonhoeffer (F/15-30, score=0.258); Nicolas Sarkozy (M/60+, score=0.260); Anne-Marie Hagelin (F/<15, score=0.262); Steven Spielberg (M/60+, score=0.262)

**Reviewer notes**:
- Relevance: ☐ rule  ☐ definition  ☐ prose  ☐ off-topic
- Feature path(s):
- Yoga draft:

---

## C15. freq=9 — `ia_Chatterjee_Advanced_Medical_Astrology.txt`
**Feature hints**: h_8, h_9, dusthana, mother_specific
**Gender split**: M=7 F=2
**Age-bucket split**: 15-30=3, 30-45=2, 60+=2, <15=2

```
When a malefic Saturn is 
associated with any of the planets causing death he over sides all others 
and himself causes death (AM, Jan 2001, P 45). Here Saturn is his 
11
111
/12'~ lord and is associated with Mercury, Ketu, Venus all malefic 
planets for his Lagna. Hence he died during Saturn-Jupiter dasa. 11 oh 
lord in a dusthana leads to daridrayoga where one gets crushed under 
debts, becomes cruel, suffers from ear troubles, get caught in criminal 
or antisocial, is a liar and becomes servile. Kuhuyoga is caused by the 
4'h lord in 6'h/8tlo/l2
111
• this makes one bereft of mother or vehicles, friends 
happiness ornaments,relations, without a house with poor health. For a 
happy life in one's mundanejumey 4
111 lord should combine with a trine 
lord to form a Raja Yoga as per Parasar. Until or unless there is a Raja 
Yoga financial stability can never be achieved. When the ascendant 
lord is with the malefic in 6
111/S'h/12
111
, bodily well being will be lacking 
(observe here also conjunction of the malefics are needed to produce a 
poor health). Such natives may suffer from poor health. 
Malefic 
planets in malefic houses generate good results. On account of this 
reason 6'h lord in 3'd/6
111/8
111/ll'h/12'b posited alone produces excellent 
(42)

--- Page 42 ---
health. Sound health should be anticipated when this planet is aspected 
by a natural benefic. We will examine: the horoscope of Prof. RJ. 
Galgali, ex Professor of Chemistry […truncated…]
```
**Sample subjects** (top 5 by score): Gelsey Kirkland (F/15-30, score=0.262); Hervé Bazin (M/30-45, score=0.263); Bill Clinton (M/<15, score=0.265); Steven Spielberg (M/60+, score=0.266); James Dean (M/60+, score=0.267)

**Reviewer notes**:
- Relevance: ☐ rule  ☐ definition  ☐ prose  ☐ off-topic
- Feature path(s):
- Yoga draft:

---

## C16. freq=9 — `P.V.R._Narasimha_Rao_-_Vedic_Astrology_Textbook.pdf`
**Feature hints**: mrityu_bhaga, h_2, h_8, h_12, maraka, eighth_lord, yogakaraka
**Gender split**: M=4 F=5
**Age-bucket split**: 15-30=4, 30-45=2, 45-60=2, <15=1

```
Venus is lagna lord and he gives Vesi yoga being in the 2nd from Sun. 
So his dasa is good. However, being the 8th lord from AL, he can also give a fall in 
status. 
Rahu’s antardasa runs during 1994-1997. As per Rath’s “tripod of life” principle, we 
should give importance to Rahu’s position from Moon. Rahu is in the 8th house from 
Moon and his antardasa is unfavorable. It can bring unexpected troubles and 
frustration. Debilitated Rahu’s rasi aspect on AL can also cast a shadow on one’s 
image and affect one’s status. 
Ketu’s pratyantardasa ran during December 1995-February 1996 when a lot of this 
drama unfolded and he was evicted from house with his accounts frozen. That shook 
him up. We should give importance to lagna when judging pratyantardasas. Ketu is 
lagna lord in the 8th house. Again it is a bad period. 
Sun’s pratyantardasa was running when he went back to India, leaving everything he 
had behind – including his job at a big company. Sun is the 10th lord and he is 
debilitated in the 12th house, afflicted by Saturn and 8th lord. Moreover Sun is in the 
8
th house from AL and that adds to the indications of a fall in status. So Sun’s 
pratyantardasa can bring losses and fall in status. During this pratyantardasa, he gave 
up everything he had and went back to India. 
Jupiter is the 8th and 11th lord from Moon. He is in the 3rd from Moon. Jupiter in 3rd
gives a positive spirit. Jupiter antardasa can inject some energy into his nearly 
destroy […truncated…]
```
**Sample subjects** (top 5 by score): Gwyneth Paltrow (F/30-45, score=0.260); Virginie Hériot (F/<15, score=0.260); Irene Forbes-Mosse (F/15-30, score=0.261); Paul Newman (M/15-30, score=0.261); Richard Boone (M/30-45, score=0.262)

**Reviewer notes**:
- Relevance: ☐ rule  ☐ definition  ☐ prose  ☐ off-topic
- Feature path(s):
- Yoga draft:

---

## C17. freq=9 — `ia_Sanjay Rath - Crux of Vedic Astrology-Timing of Events (1998)_2_djvu.txt`
**Feature hints**: eighth_lord, mother_specific
**Gender split**: M=1 F=8
**Age-bucket split**: 15-30=2, 45-60=3, <15=4

```
Timing : The demise of mother is timed from 
Matru Shook dasas. (Refer Jaimini Sutras). In this 
chart, the tenth house is stronger than the fourth and 
will initiate the matru shoola dasas. Both Mars and 
Moon are in trines in earthy signs. Thus, earthy sign 
or fiery signs in the eighth can cause death. The 
shoola dasas beginning from Virgo are for 9 years 
eac h and mothers death occurred in the first dasa 
itself. There is another method to confirm the Iongivity 
°f mother. Consider the lords of the first, eighth and 
tenth houses counted from the fourth house from 
!*gna. Here the fourth house is Pisces and the lord of 
e first and tenth Jupiter is afflicted by Rahu in a 
f Usthana from lagna. The eighth Lord Venus is also 
no ^ c * e ^ hy Mars and the nodes Rahu & Ketu. Since 
ne of the three are strong, negligible longevity is 
1Ca ec *- This method of longevity estimation finds 


138 


The Crux of Vedic Astrology-Timing of Events 


mention in most of the important works like Brihat 
Parasara Hora Shastra, Jaimini's Upadesa Sutras, Jataka 
Parijatha etc. If at least one is strong, longevity is upt 0 
36 years and if two, upto 72 years and if all three are 
strong, matrusukha even beyond 72 years of the life of 
the native. 

7.4.3. Loss of Mother-In-Law 

Chart 41, Male born on 13th November 1957 at 
5 a.m. at Delhi (28N39,77E13) 


1 

KETU 



AL 

CHART 41 

MALE 13.11 1957 

5 . Off AM 

DELHI 

MOON 

UL 

■ 

VEN 

MERC- 

SAT 

// RAHU 
' LAO SUN 
MARS 

A 4 […truncated…]
```
**Sample subjects** (top 5 by score): John F. Kennedy (M/45-60, score=0.246); Irene Forbes-Mosse (F/15-30, score=0.247); Virginie Hériot (F/<15, score=0.247); Judy Garland (F/<15, score=0.248); Fiamma Izzo (F/45-60, score=0.249)

**Reviewer notes**:
- Relevance: ☐ rule  ☐ definition  ☐ prose  ☐ off-topic
- Feature path(s):
- Yoga draft:

---

## C18. freq=8 — `Hora Sara RSanthanam Eng.pdf`
**Feature hints**: h_8, h_9, ninth_lord, father_specific, mother_specific
**Gender split**: M=1 F=7
**Age-bucket split**: 15-30=2, 30-45=2, 45-60=3, 60+=1

```
Hi s longevity, means of livelihood and death are* 
governed by Saturn. 
rRT W TO # f f^Rftcf 71^ 11 22 II 
Should the planets be devoid of strength at birth, the effects 
they generate wil l be equally weak. This does not, however, 
apply to Saturn. The reverse holds good in his case. 176 HORASARA 
Notes: Saturn's strength is essential for good longevity while 
his weakness may not give rise to debts and the like. 
STWlf ^ ^ '>#^ 11 23 II 
The effects revealed by the various planets wil l be in propor�tion to the benefic dots in the houses occupied by them. The 
results of the dasas cannot be estimated without such Ashtaka 
Varga charts. 
3?J(?ftsWR ^ Tife: fm^ F^: 11 24 II 
ds^ir^l'+.dy'tei|lM^WrwirM'J^*RI 
i^TtaW^rRro ^ ^nft ^rrj^ 11 25 11 
crfFPt ^ ftWf M ^rf^Tzrf^ ^ Wf : I 
cri^^TTFT^ ^5f q fq^: f^m¥^ ^ 11 26 II 
Wt ^ ^HlilK^lir^s^^4l^ ^1 
Take the Ashtaka Varga chart of the Sun, and find out the 
benefic dots. The Rasi which is ninth from the Rasi occupied by 
the Sun is related to one's father. The Sodhya Pinda should be 
multiplied by the said figure of benefic dots and the resultant 
figure should be divided by 27. When Saturn transits the particu�lar asterism as denoted by the remainder in the above process 
wil l cause death of the native's father. The stars in Kona position 
to the earlier mentioned star wil l also function similarly. It 
should be a Chhidra Dasa (fes ^ ) to cause such an effect. 
Notes: For detailed information regarding […truncated…]
```
**Sample subjects** (top 5 by score): Naomi Mitchison (F/30-45, score=0.244); Roberta Manfredi (F/45-60, score=0.249); Pierre Champion (M/30-45, score=0.250); Shari Belafonte (F/60+, score=0.252); Anne Donahue (F/45-60, score=0.252)

**Reviewer notes**:
- Relevance: ☐ rule  ☐ definition  ☐ prose  ☐ off-topic
- Feature path(s):
- Yoga draft:

---

## C19. freq=8 — `P.V.R._Narasimha_Rao_-_Vedic_Astrology_Textbook.pdf`
**Feature hints**: h_8, h_9, father_specific, mother_specific
**Gender split**: M=2 F=6
**Age-bucket split**: 15-30=5, 30-45=1, 45-60=1, <15=1

```
Saturn shows longevity, livelihood, fears, sadness, dangers and sorrows. 
When we want to time events relating to a particular matter, we should first fix the 
relevant planet. Then we should fix the relevant house. We should then find the 
number of rekhas in that house from that planet in that planet’s BAV. We should 
multiply it by the sodhya pinda of the planet (also called yoga pinda). By dividing 
the product with 27 or 12 and taking the remainder, we should find the associated  338 Vedic Astrology: An Integrated Approach
nakshatra or rasi. Then we can time key events based on the transits in that nakshatra 
and rasi. 
For example, suppose we want to time the good and bad periods of a native’s father. 
Father should be seen from Sun and the 9th house. We can take Sun’s BAV and find 
the number of rekhas in the 9th house from Sun. Suppose Sun is in Aq. Suppose 
Sun’s BAV contains 5 rekhas in Li (the 9th from Aq). Suppose Sun’s sodhya pinda is 
86. Multiplying 86 with 5, we get 430. If we divide 430 by 27, the quotient is 15 and 
the remainder is 25. The 25th constellation is Poorvabhadrapada. So Saturn’s transit 
in Poorvabhadrapada is bad for father and Jupiter’s transit in the same nakshatra is 
good. Now let us find the rasi. By dividing 430 by 12, we get a quotient of 35 and a 
remainder of 10. So Saturn’s transit in Cp (the 10th rasi of the zodiac) is bad for 
father and Jupiter’s transit in Cp is good. 
Though Parasara indicated that we can take a […truncated…]
```
**Sample subjects** (top 5 by score): Gwyneth Paltrow (F/30-45, score=0.247); Serge Dalens (M/45-60, score=0.248); Paul Doumer (M/<15, score=0.251); Emmi Bonhoeffer (F/15-30, score=0.254); Lise Børsum (F/15-30, score=0.258)

**Reviewer notes**:
- Relevance: ☐ rule  ☐ definition  ☐ prose  ☐ off-topic
- Feature path(s):
- Yoga draft:

---

## C20. freq=8 — `ia_Sanjay Rath - Crux of Vedic Astrology-Timing of Events (1998)_2_djvu.txt`
**Feature hints**: h_9, maraka, father_specific
**Gender split**: M=4 F=4
**Age-bucket split**: 15-30=2, 30-45=1, 45-60=3, 60+=1, <15=1

```
Jupiter 
was retrograde in 23°25' Sagittarius i.e. in Scorpio 
navamsa. This is technically called Rahu Bhava 
chandramsa as Rahu is in Sagittarius in Bhava 
(The sign occupied by transit Jupiter) and the Moon 
(chandra) is in Scorpio (the navamsa occupied 
by transit Jupiter). This brings the destructive powers 
of Rahu on ones personal affairs (Moon). Similarly, 
transit Saturn foreboding evil was on the Arudha 
Lagna (AL) and in Libra navamsa (Saturn 10°52' 
Pisces). This brings forth the transit called 
Arudha bhava Mritamsa as the navamsa transit of Sat¬ 
urn is in the eight from Arudha Lagna, and 
as explained earlier, the Mercury & Venus in eighth 
from Arudha Lagna were activated to give the 
accident. 

Thus while using Vimsottari dasas, good knowl¬ 
edge of Brighu transits is an added advantage for 
perfect timing. In the dwadasamsa (D-12 Chart). The 
ninth house (Father) is Cancer. The dasa Lord Mer¬ 
cury is evil in Scorpio being aspected by Rahu the 8th 
Lord from Cancer. Venus in the tenth can be a maraka 
for father being in 2nd from 9th house. Saturn is 7th 
Lord from Cancer placed in the 4th (Vehicles) from it 
in a Venusian sign and Ketu is in 12th from Saturn 
and with Mars indicating sudden death. Father died 
in the violent accident on 21st May 1996 in Mars-Ven- 
Sat-Ketu Vimsottari periods. Shoola dasas can also be 
used to time death. Death of father can be expected in
```
**Sample subjects** (top 5 by score): Jimmy Carter (M/15-30, score=0.244); Lise Børsum (F/15-30, score=0.244); Charles Kuwasseg (M/30-45, score=0.250); Fiamma Izzo (F/45-60, score=0.259); Georges Balagny (M/45-60, score=0.260)

**Reviewer notes**:
- Relevance: ☐ rule  ☐ definition  ☐ prose  ☐ off-topic
- Feature path(s):
- Yoga draft:

---

## C21. freq=7 — `ia_Sanjay Rath - Crux of Vedic Astrology-Timing of Events (1998)_2_djvu.txt`
**Feature hints**: h_9, eighth_lord, dusthana
**Gender split**: M=7 F=0
**Age-bucket split**: 15-30=1, 30-45=1, 45-60=2, 60+=1, <15=2

```
SUN 

MERC 

MARS 

KETU 



1 

NAVAMSA 
(D-9) CHART 


VEN 

SAT 



■ 

JUP 

RAHU 

MOON 



Sun 27°44' Moon 10°.8' Mars 12° 04’ 

Mercury 29°36' (AK) Jupiter 14°.44' Venus 12°43' 

Saturn 11°40' Rahu 20°52' 

Arudha Lagna (AL) Aries Mrityupada (A8) Pisces 
Hora Lagna 9s 9°22' 

























332 The Crux of Vedic Astrology-Timing of Events 

In Chart 87, a Nadi had made the prediction of the 
natives death in his 70th year in Sravan month (Sun in 
Cancer), Suklapaksha 14th Tithi, in Aries ascendant 
The native, however died in his 59th year in Sravan 
(Sun in Cancer), Sukla paksha 6th Tithi in Aries ascen¬ 
dant. Was the Nadi reading missing out some point? Let us 
attempt to time death. 

Kakshya : (A) (i) Lagna Lord+ 8th Lord =Fixed +Fixed = Short life 

(ii) Lagna + Hora Lagna = Dual + Movable = Short life 

(iii) Saturn + Moon = Fixed + Movable = Middle life. 

Jupiter is conjoined the lagna Lord, atmakarak 
(chara - Mercury) and natural atmakarak Sun. This 
causes Kakshya Vriddhi to middle life 36-72 years. 

(B) Lagna Lord is in dusthana (6th house), 10th 
Lord Jupiter is in dusthana (6th house) as 8th Lord 
Saturn is in trines (9th house). This indicates short life 
as only Saturn is strong.
```
**Sample subjects** (top 5 by score): Arnold Schwarzenegger (M/15-30, score=0.259); Al Gore (M/45-60, score=0.259); João Paulo Diniz (M/60+, score=0.262); Richard Boone (M/30-45, score=0.262); Paul Doumer (M/<15, score=0.265)

**Reviewer notes**:
- Relevance: ☐ rule  ☐ definition  ☐ prose  ☐ off-topic
- Feature path(s):
- Yoga draft:

---

## C22. freq=7 — `ia_Sanjay Rath - Crux of Vedic Astrology-Timing of Events (1998)_2_djvu.txt`
**Feature hints**: h_8, h_9, h_12, maraka, father_specific
**Gender split**: M=7 F=0
**Age-bucket split**: 15-30=1, 30-45=1, 45-60=2, 60+=2, <15=1

```
This is technically called Rahu Bhava 
chandramsa as Rahu is in Sagittarius in Bhava 
(The sign occupied by transit Jupiter) and the Moon 
(chandra) is in Scorpio (the navamsa occupied 
by transit Jupiter). This brings the destructive powers 
of Rahu on ones personal affairs (Moon). Similarly, 
transit Saturn foreboding evil was on the Arudha 
Lagna (AL) and in Libra navamsa (Saturn 10°52' 
Pisces). This brings forth the transit called 
Arudha bhava Mritamsa as the navamsa transit of Sat¬ 
urn is in the eight from Arudha Lagna, and 
as explained earlier, the Mercury & Venus in eighth 
from Arudha Lagna were activated to give the 
accident. 

Thus while using Vimsottari dasas, good knowl¬ 
edge of Brighu transits is an added advantage for 
perfect timing. In the dwadasamsa (D-12 Chart). The 
ninth house (Father) is Cancer. The dasa Lord Mer¬ 
cury is evil in Scorpio being aspected by Rahu the 8th 
Lord from Cancer. Venus in the tenth can be a maraka 
for father being in 2nd from 9th house. Saturn is 7th 
Lord from Cancer placed in the 4th (Vehicles) from it 
in a Venusian sign and Ketu is in 12th from Saturn 
and with Mars indicating sudden death. Father died 
in the violent accident on 21st May 1996 in Mars-Ven- 
Sat-Ketu Vimsottari periods. Shoola dasas can also be 
used to time death. Death of father can be expected in 


240 The Crux of Vedic Astrology-Timing of Events 

the month when the Sun transits the natal navamsa. 
The Sun is in Taurus navamsa in natal chart and on  […truncated…]
```
**Sample subjects** (top 5 by score): Jimi Hendrix (M/45-60, score=0.242); Roger Boutet de Monvel (M/30-45, score=0.248); Nicolas Sarkozy (M/60+, score=0.249); Arnold Schwarzenegger (M/15-30, score=0.253); Robert Redford (M/45-60, score=0.254)

**Reviewer notes**:
- Relevance: ☐ rule  ☐ definition  ☐ prose  ☐ off-topic
- Feature path(s):
- Yoga draft:

---

## C23. freq=7 — `ia_Book. Encyclopedia of vedic astro - dasha systems.pdf`
**Feature hints**: dasha_chain, father_specific
**Gender split**: M=4 F=3
**Age-bucket split**: 30-45=1, 45-60=2, 60+=1, <15=3

```
Death may be caused by the lords of the 6th, 8* and the 12* houses in 
the course of the Major Period of the 8* lord who occupies the 6th, the 
8* orthe 12*. 
If the lords of the 1 0* and the 3'd are in conjunction with, or aspect 
each other, the native will be deprived of fortune in the Major Period of 
the 1 0* lord and he will enjoy fortunate results in the Major Period of 
the 3'd lord. Chapter 02: Vimsottari Oasa System of Predictions 69 
If the 51
h, the 7* and the 9* lords are in their own houses, they give 
rise to dips in Ganga during their periods and sub-periods. 
A person gets wealth by his own exertions in the Major Period of a 
planet occupying the Ascendant or the 7* house. He will also acquire 
much wealth by his own efforts in the Major Period of the 9* lord 
occupying the 7* house. 
The native's father will die in the sub-period of Rahu, Ketu, Saturn or 
the Sun within the Major Period of Rahu. 
Father's death may be predicted iathe sub-period of Mars, Saturn, the 
Sun or Rahu in the Major Period of Ketu. 
Father's death will happen in the sub-periods of Rahu, Ketu, Saturn or 
the Sun in the Major Period of Mars. 
Father will die in the sub-period of Rahu, Mars, the Sun or Ketu in the 
Major Period of Saturn. 
Father's death will certainly happen just before the end of Mars Major 
Period and the beginning of Rahu Major Period. 
Astrologers say that one's father will die in the sub-period of Rahu in 
the Major Period of a malefic pla […truncated…]
```
**Sample subjects** (top 5 by score): Sylvia Plath (F/<15, score=0.250); John F. Kennedy (M/45-60, score=0.253); Virginie Hériot (F/<15, score=0.253); Laurent de Brunhoff (M/<15, score=0.254); George W. Bush (M/60+, score=0.261)

**Reviewer notes**:
- Relevance: ☐ rule  ☐ definition  ☐ prose  ☐ off-topic
- Feature path(s):
- Yoga draft:

---

## C24. freq=7 — `ia_MN_Kedar_Predictive_Astrology.txt`
**Feature hints**: h_7, h_8, h_9, maraka, father_specific, mother_specific
**Gender split**: M=4 F=3
**Age-bucket split**: 30-45=4, 45-60=1, <15=2

```
10.22 AQUARIUS
Jupiter, the Moon and Mars are malefics. Venus and
Saturn are auspicious. 
Venus is the only strongest
Raja yoga. karaka (being lord of 4th and 9th). Jupiter,
the Sun and Mars may act as killers. Mercury is
medium (may give mixed results). 
Jupiter (lord of 2nd
and 11th houses), the Moon (6th lord) and Mars(lord
of 3rd and 10th) are malef ics (qrw:).
1 5 9


--- Page 150 ---
Predictive Astrology
10.23 PTSCES
Mars with Moon or Jupiter causes Rajayoga. Moon
and Mars are auspicious. 
Mars is the most powerf ul.
Mars is a killer too, but will not kill of his own, being
also lord of Trine house (9th house). Mars does not
kill of his own unless instigated by another killer
Saturn or Mercury. 
Saturn, Sun, Venus and Mercury
are evils. Saturn is the lord of 11th and 12th, both
evil houses. Venus lord of 3rd and 8th is also lord of
two evil houses. Sun ruling 6th house is also an
inasupicious 
planet.
1.
2.
3.
4.
Question
Explain what do you understand 
by the term "Key
planets"?
Def ine Kendradhipatya 
Dosha.
What are the planets causing obstruction for
movable, fixed and dual sign lagnas?
Write short notes on:
(a) Trishadayadhipati
(b) Lords of 8th and 12th houses
(c) Role of Kendra and Trikona lords.
Why any Planet becomes auspicious or
inauspicious 
for a particular 
lagna ?
5.


--- Page 151 ---
11
AYURDAYA AND MARAKA
11.1 The two topics will be discussed 
one by one.
We have already discussed in the previous […truncated…]
```
**Sample subjects** (top 5 by score): Roberta Manfredi (F/45-60, score=0.259); Mason Alan Dinehart (M/<15, score=0.262); Aretha Franklin (F/30-45, score=0.263); Gianni Agnelli (M/<15, score=0.264); Pierre Champion (M/30-45, score=0.264)

**Reviewer notes**:
- Relevance: ☐ rule  ☐ definition  ☐ prose  ☐ off-topic
- Feature path(s):
- Yoga draft:

---

## C25. freq=6 — `ia_Ashtakavarga_C.S. Patel_1957 ed_djvu.txt`
**Feature hints**: h_9, father_specific
**Gender split**: M=4 F=2
**Age-bucket split**: 15-30=1, 30-45=1, 45-60=3, <15=1

```
Saharan Raa werent | 

WMASMISAWSTT AAT AW AAR AT? Il ¢ I 
Sloka 8— When the Sun, associated with 6, 5, 7 or 8 
bindus, occupies a Kendra or Trikona position, the native 
or his father will meet with death in 22nd, 35th, 30th or 
36th year respectively of the native. 
Notes: Devakeralam goes a little further and says that in the 
same circumstances the person or his father will meet with an 
accidental or immediate death, i.e., by fire or by a fall from a 
mountain or in a cemetery, etc. 


nara tahaa aati | 
Wal ssa: Aral Tae USTMTETN: (1 SU 


Sloka 9 — When the Sun, in conjunction with the Moon, 
Saturn and Mercury, is in a Kendra at birth and that bhava 
has 2 nett bindus, i.e., after both the reductions, persons 
learned in Astrology say that, the father of the native will 
have immense administrative power and fame ten years 
after the native’s birth. 


THY TAT Tal sz aT waa 
REIT fg AIA AW I Lo Ul 


Sloka 10—A person born having Rahu, Saturn or Mars. 
in the 4th house from the Sun, will be the cause of his 
father’s early death provided that house (4th house from 
the Sun) is not aspected either by Jupiter or Venus. 


BARTER Te TATA aly | 

Reareay cet are Afra aeetae 1 22 II 

MIFOMSA ARIAT: | 
Slokas 11-114 — When Saturn transits the 9th house from 
the Lagna or the Moon, the demise of one’s parents may be 
predicted, if he (Saturn) is aspected by or associated with 
a malefic planet and also the Dasa and Antardasa at that 
time indicate the same. . = 


6 aTeRaT: […truncated…]
```
**Sample subjects** (top 5 by score): Tennessee Williams (M/45-60, score=0.255); Eugenio Garza Lagüera (M/45-60, score=0.258); Melanie Griffith (F/30-45, score=0.261); Adolphe, Grand Duke of Luxembourg (M/15-30, score=0.263); Anne-Marie Hagelin (F/<15, score=0.264)

**Reviewer notes**:
- Relevance: ☐ rule  ☐ definition  ☐ prose  ☐ off-topic
- Feature path(s):
- Yoga draft:

---

## C26. freq=5 — `ia_Ashtakavarga_C.S. Patel_1957 ed_djvu.txt`
**Feature hints**: h_9, father_specific, mother_specific
**Gender split**: M=2 F=3
**Age-bucket split**: 30-45=1, 45-60=1, <15=3

```
Sloka 10—A person born having Rahu, Saturn or Mars. 
in the 4th house from the Sun, will be the cause of his 
father’s early death provided that house (4th house from 
the Sun) is not aspected either by Jupiter or Venus. 


BARTER Te TATA aly | 

Reareay cet are Afra aeetae 1 22 II 

MIFOMSA ARIAT: | 
Slokas 11-114 — When Saturn transits the 9th house from 
the Lagna or the Moon, the demise of one’s parents may be 
predicted, if he (Saturn) is aspected by or associated with 
a malefic planet and also the Dasa and Antardasa at that 
time indicate the same. . = 


6 aTeRaT: cnap. VI 


sareaaranacarat + pera 2 
gaaracaa F Ted TAT I 83 Il 
Slokas 12-13-— The demise of the father may occur also 
during the Dasa of the planet who is the lord of the rasé 
occupied by the lord of the 4th house from the Lagna, or 
it may happen probably during the Dasa of the lord of the 
4th house. 
Devakeralam and other works :— 

1. When the Sun is in a Kendra and even when that 
is a friendly house, associated with 3, 4 or 5 bindus, the 
father of the native will meet with death or suffer afflic- 
tions in the 17th year of the native. 

2. When the Sun is in the 5th or 9th bhava, the 
father of the person will be afflicted by misfortune at the 
age (of the person) represented by the number of bindus 
in that house in the Samudayashtakavarga. 

3. If the Sun is in the 2nd or 5th house associated 
with 3 bivdus and Rahu is in the 9th, the person will be 
-bereft of his father at his 5th year of age […truncated…]
```
**Sample subjects** (top 5 by score): Tennessee Williams (M/45-60, score=0.252); Barbra Streisand (F/<15, score=0.261); Bernt Heiberg (M/30-45, score=0.262); Judy Garland (F/<15, score=0.265); Marlene Dietrich (F/<15, score=0.267)

**Reviewer notes**:
- Relevance: ☐ rule  ☐ definition  ☐ prose  ☐ off-topic
- Feature path(s):
- Yoga draft:

---

## C27. freq=3 — `ia_Medical_Astrology_Krishna_Kumar.txt`
**Feature hints**: h_8, h_9, eighth_lord, ninth_lord, father_specific
**Gender split**: M=2 F=1
**Age-bucket split**: <15=3

```
Ultimately, Astrology lies in the fact that a native 
is destined to have his kith and kin in a definite 
fashion as per the ordain of the planets on his kith 
and kin imposing diseases, accidents etc. or causing 
happiness to them through marriage, birth of sons 
etc. 
To narrate in simple terms, it is destined for a 
native to be the son or daughter of a heart patient, 
diabetic patient, parents with different dieases like 


--- Page 136 ---
Medical Astrology 
143 
leukoderma, short of hearing, eye problems etc. or as 
children of healthy parents. 
It is also destined that a native may have healthy 
co-borns or no-borns at all, sickly or healthy wife etc. 
15. DIAGNOSIS OF DISEASES OF KITH AND KIN FROM 
A GIVEN CHART 
(i) 
Diagnosis of Disease of Father 
~upiter 
Moon 
Rahu 
Chart XXI 
16-11-1951 
Ketu 
MercUIJ 
Lagna 
Sun Sat Ver 
Mars 
Saturn 
Venus Lagna 
Sun 
Navamsha 
Ketu 
Mars 
Chart 
Moon 
Rahu 
Merc 
Jupiter 
In the chart XXI, ninth lord Venus representing 
the house of father is debilitated and is conjoined with 
6th and 8th lord. This combination is evident to show 
some trouble to the father by way of disturbance in 
health. 


--- Page 137 ---
144 
Medical Astrology 
To diagnose exactly the dise~se of the father, the 
6th, from the 9th house Libra is occupied by a 
debilitated planet Sun who is the 4th lord of the 
father's house and represents the region of the heart 
of the father. Sun being debilitate […truncated…]
```
**Sample subjects** (top 5 by score): Albert Camus (M/<15, score=0.264); Laurent de Brunhoff (M/<15, score=0.266); Sylvia Plath (F/<15, score=0.270)

**Reviewer notes**:
- Relevance: ☐ rule  ☐ definition  ☐ prose  ☐ off-topic
- Feature path(s):
- Yoga draft:

---

## C28. freq=2 — `Sarvarth Chintamani Bhasin of Vyankatesh Sharama Bhashin J.L. (Astrology).pdf`
**Feature hints**: h_8, h_9, father_specific, mother_specific
**Gender split**: M=1 F=1
**Age-bucket split**: 30-45=1, <15=1

```
in mo>uhlc sign•. the son will 
not have opportunity to light the pyre of the parents. If they are in 
kcndra & in common sign. there will be delay in the lighting oft he 
pyre>. (55) 
~ ~ ft1qqi<Hiq ¥1*1<t>~S~'<Iw~~i'41 : I 
~ !RJ ~ d\l\ll'1"41 ~ ~ ('t~«IS{rtl II'(& II 
If the lord of the 9th house is strong and Moon. lord uflugna and 
the lord of the 4th house arc without ~trcngth. the yoga indicate' 
severe trouble to the mother. (56} 
190 
\o"'l' q QIQill~cM ~ fllf.l~~~~ t 
~~:t'~\1?1~ 11'(1911 
If there arc many malefic planets in the 8th hou,e. disease on or 
near the r.:ctum should be declared. not however. when there i~ 
bcnellc influence on the 8th house. (57) 
{"'] ~ t-'1 ~ • 'litq I 
~~~~11'(<;11 
The native leads a comfortable life. if the 8th house i' occupied 
t>> ,)r aspected by b.:nclics. If there is influence of male tics on the 
Sth. one eats coarse food: otherwise. relined t)pe. "(58) 
~qtQE)~~~I 
~l.l[{"'];rnr: ~: 11'(0.11 
DEFEAT 
If the lord of the 8th house i• located with malelics. is aspected 
by malefic;. is located in bcrwcen maleli.:s or if the ;ign of a 
malellc is in the 8th house. the nati'c often meets defeat in his 
undertakings. (59) 
~ ~ {"'] 'lft"'<h fW I . 
'f(tli'~m~ tt&ott 
lfthcrc is influence of benefic planets on the 81h house there is 
no lie feat for the native. If the sign in the 8th house is a movable 
one. & the lord of the 8th is also in a mo,ahle sign the native die~ 
away from hi> home. (60) 
A […truncated…]
```
**Sample subjects** (top 5 by score): Ray Bradbury (M/30-45, score=0.257); Abigail Kapiolani Kawānanakoa (F/<15, score=0.268)

**Reviewer notes**:
- Relevance: ☐ rule  ☐ definition  ☐ prose  ☐ off-topic
- Feature path(s):
- Yoga draft:

---

## C29. freq=2 — `ia_J_KP reader_3_Predictive Stellar Astrology_djvu.txt`
**Feature hints**: h_9, father_specific, mother_specific
**Gender split**: M=2 F=0
**Age-bucket split**: 30-45=1, 45-60=1

```
Thus one should note separately what each planet signifies. 
Then in which sub each planet is posited. 


Finally one should take each house and judge, to which 
bhavas the planet are the significators and the sub lords. Judge 





KRISHNAMURTI PADHDHATI 145 





calmly how each bhava is receiving good results from a few and 
adverse from the rest. 


Say, mother, father, child, etc. Mother is shown by 4th house. 
Father by 9th house. child by 5th house. 


IX 0-30 
Jup. 8-27 Moon 12-25 XI 1-30 
VII 28-30 gtaaa 
VII 0-04 29-1-1928 
7-06 P M. 


13-04 N Il 28-30 
Mer. 29-2 
Sun 15-40 80-15 
VI 0-30 : 
Mars 14-59 Sat 23-37 
Ven. 7-53 |Kethu 23-23] 1410-20 
V 1-30 IV 1-30 


pic 
So, note the significators of 4: so also the significators of 9 
and 5. 





Let us take the above chart. 

Kethu Qasa balance 0-5- 

23, 

The constellation ruled by a planet indicates the matters, 
signified by the bhava occupied or owned by it. 


The sub lord occupied by a planet denotes whether it is 
auspicious for the progress of that Bhava or inauspicious so that 
one has obstacle or faces disappointment or negation of the 
matter. 





146 * PREDICTIVE STELLAR ASTROLOGY* 





This applies to all bhava results. If lagna is occupied by a 
planet or owned by one and if a planet either the same 
lagnadhipathi or any other planet is deposited in the occupants 
or owner's star, then they indicate first house matters. the 
depositor in that constellation is in the favourable sub matters 
indicate […truncated…]
```
**Sample subjects** (top 5 by score): Roger Boutet de Monvel (M/30-45, score=0.265); Georges Balagny (M/45-60, score=0.266)

**Reviewer notes**:
- Relevance: ☐ rule  ☐ definition  ☐ prose  ☐ off-topic
- Feature path(s):
- Yoga draft:

---

## C30. freq=2 — `ia_Ashtakavarga_C.S. Patel_1996 ed_djvu.txt`
**Feature hints**: h_8, h_9, father_specific, mother_specific
**Gender split**: M=1 F=1
**Age-bucket split**: 60+=1, <15=1

```
g Renfesrantagetacect | 
wared frais Rrewnsys fronfeqgana: ret ws aT 11 1 
Sloka 8 - In order to find the time of death of the father, the 
mother and other relatives, multiply the Shodhyapinda of the 
Karaka planet (as referred to in the previous verse) by the 
number of bindus (before the reductions) in the relevant bhava 
reckoned from the position of the Karaka planet and divide the 
product by 27. Saturn's transit through the asterism represented 
by the remainder counted from Aswini, brings about the death of 
the respective person. 
¢, Beeesatr | 


168 wea: 


Notes :In the previous verse. Shodhyavashishta is taken into 
consideration, while in this verse Shodhyapinda is multiplied. 


In the Standard Horoscope multiply the Shodhyapinda of 
the Sun 204 x 4 (the number of bindus in the 9th house from 
the Sun before the reductions in the Sun's Ashtakavarga) = 816. 
Diving by 27 the remainder is 6, i.e., Ardra. 


Tega Wet aaarorah Recarenyq | 
HPA THAT TAT ACTA : 118 


Sloka 9 - When Rahu is in either Dhanus or Meena (in the natal 
chart) and Jupiter transits that rasi or its trines, that period will 
bring misfortune to the person, or if Rahu and Jupiter are in 
conjunction in a natal chart and when Jupiter transits that rast or 
its trines, that period will bring misfortune to the person. 


When Jupiter trasits the asterism occupied by the lord of 
the 6th house (or its trines), there is fear of death. When the 
same asterism (or its trines) is transitted by Satum  […truncated…]
```
**Sample subjects** (top 5 by score): Michel Fugain (M/60+, score=0.253); Lucille Ball (F/<15, score=0.255)

**Reviewer notes**:
- Relevance: ☐ rule  ☐ definition  ☐ prose  ☐ off-topic
- Feature path(s):
- Yoga draft:

---

## C31. freq=2 — `ia_Analysing Horoscope through modern techniques_Mehta_djvu.txt`
**Feature hints**: h_2, h_8, h_9, father_specific
**Gender split**: M=0 F=2
**Age-bucket split**: 30-45=2

```
FEMALE 


Horoscope 2 

However, Jupiter as spiritual planet in 8th Moksha Trikona can be 
extremely good for yogic practice and achieving immortality, as 
horoscope of Adhishankar Acharya shows : 


160 Analysing Horoscope Through Modern Techniques 


Sun 
Mars Mer Moun 
Ven 
| Adhi Shankar 
= ee = 


Venus 

As a natural benefic its position in eight is not good excepting a 
benefic in the house of longevity gives a long life and as a benefic 
its aspect on 2nd house of wealth is good, where it also forms part 
of adhi Yoga. But as Karka for marriage its relegation to eight is not 
beneficial for a happy married life. Venus afflicted by Saturn may 
cause impotency and unclean habits. 

Favourable: long life, fortune through cattle and attendants. 
Prosperous and rich as its aspects 2nd, profit after hard effort, gain 
and loss by turn and delayed victory. Land lord. 

Pays off his father's debts, death in a holy place. 

Unfavourable: Unpleasant speech, mean and unbecoming action. 
Sick, quarrelsome and poor. If afflicted kidney problem, diabetes, 
obesity or urinary problem and sexual diseases. Death of marriage 
partner. In Gemini or Scorpio strife with wife and children. In Aries 
or Virgo misfortunes during marriage, financial and economic crisis. 
Drug addiction in Cancer, Leo and Pisces or Scorpio. 
Venus-Rahu in Aries in Kendra to Mars gives death due to 
malignancy. (See horoscope of Nargis Dutt). 

Saturn 

As a natural malefic it is not good in the 8th house. Its  […truncated…]
```
**Sample subjects** (top 5 by score): Melanie Griffith (F/30-45, score=0.256); Alice Munro (F/30-45, score=0.267)

**Reviewer notes**:
- Relevance: ☐ rule  ☐ definition  ☐ prose  ☐ off-topic
- Feature path(s):
- Yoga draft:

---

## C32. freq=2 — `ia_Bhrigu_Nandi_Nadi_RG_Rao.txt`
**Feature hints**: h_2, h_7, h_9, h_12, father_specific, mother_specific
**Gender split**: M=0 F=2
**Age-bucket split**: 45-60=1, <15=1

```
--- Page 161 ---
166 
BHRIGU NANDI NADI 
The lord of the sixth house from Jupiter is in the fifth house with the lord of 
that house. This clearly indicates that there lSno doubt regarding unhappiness 
in matrimony. 
The significator 'of father, the Sun is ruling in his own house having 
Mercury, Venus and Rahu in the second house to him. Rahu indicates mepial 
service. Mercury indicates writing work and Venus indicates secret service. 
As Mercury is exalted the writing talent would be of a high order. The 
combination of Mercury and Venus indicates auspicious affairs. As Mercury 
and Venus are in the fifth house from Jupiter educational institutions are 
indicated. So, the native's father will have a profession related to all the 
indications mentioned above. 
Analysing in a different manner, it can be stated that Venus indicates 
beauty, Mercury indicates writing and things pertaining to intelligence and 
Rahu indicates dark colour. Therefore, the native's father will do some 
business or work related to these things. Mercury is also a significator of 
brothers. So, the native's father will have a brother (younger.) 
The Sun has debilitated· Venus in the next house to him and also 
Me1.::ury and Rahu. The succeeding houses from Mercury are vacant till we 
reach Pisces which is occupied by Ketu. This indicates that the two brothers 
will not have many children. But, Venus confers a sister to the native's father. 
Disciple: The significator of brothers  […truncated…]
```
**Sample subjects** (top 5 by score): Marlene Dietrich (F/<15, score=0.265); Jennifer Ehle (F/45-60, score=0.265)

**Reviewer notes**:
- Relevance: ☐ rule  ☐ definition  ☐ prose  ☐ off-topic
- Feature path(s):
- Yoga draft:

---