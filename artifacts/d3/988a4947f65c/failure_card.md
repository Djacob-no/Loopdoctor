RESULTS OF THE CURRENT NETWORK (256 episodes per row, actions sampled from the policy, 200 steps each).

| Layouts | Episodes with any delivery | Items delivered | Orders completed | Score | Loop index | Idle or bump fraction | Full but not delivering |
|---|---|---|---|---|---|---|---|
| T0 (the layout above) | 0.0% | 0.00 | 0.00 | 0.00 | 0.862 | 0.352 | 0.809 |
| open (normal rooms) | 98.4% | 21.07 | 5.71 | 49.61 | 0.314 | 0.103 | 0.098 |

Loop index: fraction of steps whose (cell, carried items) state already occurred in the previous 20 steps. Idle or bump: fraction of steps that were a no-op or a move into a wall or rack. Full but not delivering: fraction of steps carrying 3 items, at least one of them deliverable, with no delivery in the last 20 steps.

ONE RECORDED EPISODE ON T0 (all 200 steps; final score 0).

step | action | robot (row, col) after the step | carrying | score
   0 | start  | (1, 5) | - | 0
   1 | down   | (2, 5) | - | 0
   2 | noop   | (2, 5) | - | 0
   3 | drop   | (2, 5) | - | 0
   4 | noop   | (2, 5) | - | 0
   5 | drop   | (2, 5) | - | 0
   6 | right  | (2, 6) | - | 0
   7 | down   | (2, 6) | - | 0
   8 | left   | (2, 5) | - | 0
   9 | left   | (2, 4) | - | 0
  10 | noop   | (2, 4) | - | 0
  11 | noop   | (2, 4) | - | 0
  12 | down   | (2, 4) | - | 0
  13 | right  | (2, 5) | - | 0
  14 | left   | (2, 4) | - | 0
  15 | right  | (2, 5) | - | 0
  16 | drop   | (2, 5) | - | 0
  17 | drop   | (2, 5) | - | 0
  18 | drop   | (2, 5) | - | 0
  19 | left   | (2, 4) | - | 0
  20 | left   | (2, 3) | - | 0
  21 | left   | (2, 2) | - | 0
  22 | pick   | (2, 2) | A | 0
  23 | pick   | (2, 2) | A A | 0
  24 | right  | (2, 3) | A A | 0
  25 | right  | (2, 4) | A A | 0
  26 | noop   | (2, 4) | A A | 0
  27 | down   | (2, 4) | A A | 0
  28 | noop   | (2, 4) | A A | 0
  29 | left   | (2, 3) | A A | 0
  30 | left   | (2, 2) | A A | 0
  31 | noop   | (2, 2) | A A | 0
  32 | down   | (2, 2) | A A | 0
  33 | right  | (2, 3) | A A | 0
  34 | drop   | (2, 3) | A A | 0
  35 | down   | (2, 3) | A A | 0
  36 | right  | (2, 4) | A A | 0
  37 | up     | (1, 4) | A A | 0
  38 | down   | (2, 4) | A A | 0
  39 | right  | (2, 5) | A A | 0
  40 | left   | (2, 4) | A A | 0
  41 | noop   | (2, 4) | A A | 0
  42 | right  | (2, 5) | A A | 0
  43 | noop   | (2, 5) | A A | 0
  44 | left   | (2, 4) | A A | 0
  45 | noop   | (2, 4) | A A | 0
  46 | noop   | (2, 4) | A A | 0
  47 | up     | (1, 4) | A A | 0
  48 | down   | (2, 4) | A A | 0
  49 | pick   | (2, 4) | A A C | 0
  50 | left   | (2, 3) | A A C | 0
  51 | left   | (2, 2) | A A C | 0
  52 | left   | (2, 1) | A A C | 0
  53 | down   | (3, 1) | A A C | 0
  54 | down   | (4, 1) | A A C | 0
  55 | right  | (4, 1) | A A C | 0
  56 | down   | (5, 1) | A A C | 0
  57 | down   | (6, 1) | A A C | 0
  58 | noop   | (6, 1) | A A C | 0
  59 | down   | (7, 1) | A A C | 0
  60 | noop   | (7, 1) | A A C | 0
  61 | right  | (7, 2) | A A C | 0
  62 | pick   | (7, 2) | A A C | 0
  63 | up     | (7, 2) | A A C | 0
  64 | drop   | (7, 2) | A A C | 0
  65 | down   | (8, 2) | A A C | 0
  66 | down   | (8, 2) | A A C | 0
  67 | left   | (8, 1) | A A C | 0
  68 | right  | (8, 2) | A A C | 0
  69 | left   | (8, 1) | A A C | 0
  70 | pick   | (8, 1) | A A C | 0
  71 | down   | (8, 1) | A A C | 0
  72 | drop   | (8, 1) | A A C | 0
  73 | pick   | (8, 1) | A A C | 0
  74 | left   | (8, 1) | A A C | 0
  75 | left   | (8, 1) | A A C | 0
  76 | noop   | (8, 1) | A A C | 0
  77 | drop   | (8, 1) | A A C | 0
  78 | up     | (7, 1) | A A C | 0
  79 | left   | (7, 1) | A A C | 0
  80 | drop   | (7, 1) | A A C | 0
  81 | pick   | (7, 1) | A A C | 0
  82 | drop   | (7, 1) | A A C | 0
  83 | pick   | (7, 1) | A A C | 0
  84 | right  | (7, 2) | A A C | 0
  85 | down   | (8, 2) | A A C | 0
  86 | down   | (8, 2) | A A C | 0
  87 | down   | (8, 2) | A A C | 0
  88 | noop   | (8, 2) | A A C | 0
  89 | up     | (7, 2) | A A C | 0
  90 | noop   | (7, 2) | A A C | 0
  91 | up     | (7, 2) | A A C | 0
  92 | down   | (8, 2) | A A C | 0
  93 | up     | (7, 2) | A A C | 0
  94 | up     | (7, 2) | A A C | 0
  95 | drop   | (7, 2) | A A C | 0
  96 | pick   | (7, 2) | A A C | 0
  97 | drop   | (7, 2) | A A C | 0
  98 | pick   | (7, 2) | A A C | 0
  99 | noop   | (7, 2) | A A C | 0
 100 | pick   | (7, 2) | A A C | 0
 101 | noop   | (7, 2) | A A C | 0
 102 | down   | (8, 2) | A A C | 0
 103 | pick   | (8, 2) | A A C | 0
 104 | pick   | (8, 2) | A A C | 0
 105 | left   | (8, 1) | A A C | 0
 106 | drop   | (8, 1) | A A C | 0
 107 | pick   | (8, 1) | A A C | 0
 108 | up     | (7, 1) | A A C | 0
 109 | right  | (7, 2) | A A C | 0
 110 | drop   | (7, 2) | A A C | 0
 111 | noop   | (7, 2) | A A C | 0
 112 | up     | (7, 2) | A A C | 0
 113 | down   | (8, 2) | A A C | 0
 114 | down   | (8, 2) | A A C | 0
 115 | pick   | (8, 2) | A A C | 0
 116 | pick   | (8, 2) | A A C | 0
 117 | drop   | (8, 2) | A A C | 0
 118 | left   | (8, 1) | A A C | 0
 119 | down   | (8, 1) | A A C | 0
 120 | left   | (8, 1) | A A C | 0
 121 | down   | (8, 1) | A A C | 0
 122 | left   | (8, 1) | A A C | 0
 123 | right  | (8, 2) | A A C | 0
 124 | down   | (8, 2) | A A C | 0
 125 | drop   | (8, 2) | A A C | 0
 126 | left   | (8, 1) | A A C | 0
 127 | left   | (8, 1) | A A C | 0
 128 | right  | (8, 2) | A A C | 0
 129 | down   | (8, 2) | A A C | 0
 130 | down   | (8, 2) | A A C | 0
 131 | drop   | (8, 2) | A A C | 0
 132 | left   | (8, 1) | A A C | 0
 133 | drop   | (8, 1) | A A C | 0
 134 | left   | (8, 1) | A A C | 0
 135 | drop   | (8, 1) | A A C | 0
 136 | right  | (8, 2) | A A C | 0
 137 | pick   | (8, 2) | A A C | 0
 138 | left   | (8, 1) | A A C | 0
 139 | left   | (8, 1) | A A C | 0
 140 | right  | (8, 2) | A A C | 0
 141 | down   | (8, 2) | A A C | 0
 142 | left   | (8, 1) | A A C | 0
 143 | noop   | (8, 1) | A A C | 0
 144 | right  | (8, 2) | A A C | 0
 145 | up     | (7, 2) | A A C | 0
 146 | noop   | (7, 2) | A A C | 0
 147 | down   | (8, 2) | A A C | 0
 148 | noop   | (8, 2) | A A C | 0
 149 | up     | (7, 2) | A A C | 0
 150 | noop   | (7, 2) | A A C | 0
 151 | down   | (8, 2) | A A C | 0
 152 | pick   | (8, 2) | A A C | 0
 153 | drop   | (8, 2) | A A C | 0
 154 | drop   | (8, 2) | A A C | 0
 155 | noop   | (8, 2) | A A C | 0
 156 | drop   | (8, 2) | A A C | 0
 157 | pick   | (8, 2) | A A C | 0
 158 | pick   | (8, 2) | A A C | 0
 159 | noop   | (8, 2) | A A C | 0
 160 | up     | (7, 2) | A A C | 0
 161 | noop   | (7, 2) | A A C | 0
 162 | up     | (7, 2) | A A C | 0
 163 | noop   | (7, 2) | A A C | 0
 164 | up     | (7, 2) | A A C | 0
 165 | up     | (7, 2) | A A C | 0
 166 | drop   | (7, 2) | A A C | 0
 167 | drop   | (7, 2) | A A C | 0
 168 | down   | (8, 2) | A A C | 0
 169 | pick   | (8, 2) | A A C | 0
 170 | up     | (7, 2) | A A C | 0
 171 | left   | (7, 1) | A A C | 0
 172 | right  | (7, 2) | A A C | 0
 173 | up     | (7, 2) | A A C | 0
 174 | up     | (7, 2) | A A C | 0
 175 | noop   | (7, 2) | A A C | 0
 176 | drop   | (7, 2) | A A C | 0
 177 | left   | (7, 1) | A A C | 0
 178 | right  | (7, 2) | A A C | 0
 179 | left   | (7, 1) | A A C | 0
 180 | pick   | (7, 1) | A A C | 0
 181 | left   | (7, 1) | A A C | 0
 182 | pick   | (7, 1) | A A C | 0
 183 | up     | (6, 1) | A A C | 0
 184 | drop   | (6, 1) | A A C | 0
 185 | left   | (6, 1) | A A C | 0
 186 | right  | (6, 1) | A A C | 0
 187 | drop   | (6, 1) | A A C | 0
 188 | right  | (6, 1) | A A C | 0
 189 | noop   | (6, 1) | A A C | 0
 190 | pick   | (6, 1) | A A C | 0
 191 | left   | (6, 1) | A A C | 0
 192 | pick   | (6, 1) | A A C | 0
 193 | down   | (7, 1) | A A C | 0
 194 | down   | (8, 1) | A A C | 0
 195 | drop   | (8, 1) | A A C | 0
 196 | left   | (8, 1) | A A C | 0
 197 | left   | (8, 1) | A A C | 0
 198 | down   | (8, 1) | A A C | 0
 199 | right  | (8, 2) | A A C | 0
 200 | down   | (8, 2) | A A C | 0
