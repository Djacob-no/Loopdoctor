RESULTS OF THE CURRENT NETWORK (256 episodes per row, actions sampled from the policy, 200 steps each).

| Layouts | Episodes with any delivery | Items delivered | Orders completed | Score | Loop index | Idle or bump fraction | Full but not delivering |
|---|---|---|---|---|---|---|---|
| T0 (the layout above) | 0.8% | 0.04 | 0.01 | 0.07 | 0.854 | 0.345 | 0.858 |
| open (normal rooms) | 99.2% | 18.95 | 5.04 | 44.12 | 0.348 | 0.105 | 0.098 |

Loop index: fraction of steps whose (cell, carried items) state already occurred in the previous 20 steps. Idle or bump: fraction of steps that were a no-op or a move into a wall or rack. Full but not delivering: fraction of steps carrying 3 items, at least one of them deliverable, with no delivery in the last 20 steps.

ONE RECORDED EPISODE ON T0 (all 200 steps; final score 0).

step | action | robot (row, col) after the step | carrying | score
   0 | start  | (1, 5) | - | 0
   1 | left   | (1, 4) | - | 0
   2 | noop   | (1, 4) | - | 0
   3 | left   | (1, 3) | - | 0
   4 | noop   | (1, 3) | - | 0
   5 | down   | (2, 3) | - | 0
   6 | left   | (2, 2) | - | 0
   7 | left   | (2, 1) | - | 0
   8 | down   | (3, 1) | - | 0
   9 | right  | (3, 1) | - | 0
  10 | noop   | (3, 1) | - | 0
  11 | noop   | (3, 1) | - | 0
  12 | noop   | (3, 1) | - | 0
  13 | pick   | (3, 1) | A | 0
  14 | left   | (3, 1) | A | 0
  15 | pick   | (3, 1) | A A | 0
  16 | down   | (4, 1) | A A | 0
  17 | right  | (4, 1) | A A | 0
  18 | pick   | (4, 1) | A A A | 0
  19 | down   | (5, 1) | A A A | 0
  20 | down   | (6, 1) | A A A | 0
  21 | right  | (6, 1) | A A A | 0
  22 | down   | (7, 1) | A A A | 0
  23 | right  | (7, 2) | A A A | 0
  24 | left   | (7, 1) | A A A | 0
  25 | right  | (7, 2) | A A A | 0
  26 | up     | (7, 2) | A A A | 0
  27 | up     | (7, 2) | A A A | 0
  28 | noop   | (7, 2) | A A A | 0
  29 | down   | (8, 2) | A A A | 0
  30 | left   | (8, 1) | A A A | 0
  31 | up     | (7, 1) | A A A | 0
  32 | left   | (7, 1) | A A A | 0
  33 | left   | (7, 1) | A A A | 0
  34 | drop   | (7, 1) | A A A | 0
  35 | down   | (8, 1) | A A A | 0
  36 | pick   | (8, 1) | A A A | 0
  37 | down   | (8, 1) | A A A | 0
  38 | down   | (8, 1) | A A A | 0
  39 | pick   | (8, 1) | A A A | 0
  40 | right  | (8, 2) | A A A | 0
  41 | noop   | (8, 2) | A A A | 0
  42 | pick   | (8, 2) | A A A | 0
  43 | noop   | (8, 2) | A A A | 0
  44 | pick   | (8, 2) | A A A | 0
  45 | noop   | (8, 2) | A A A | 0
  46 | noop   | (8, 2) | A A A | 0
  47 | up     | (7, 2) | A A A | 0
  48 | up     | (7, 2) | A A A | 0
  49 | drop   | (7, 2) | A A A | 0
  50 | left   | (7, 1) | A A A | 0
  51 | down   | (8, 1) | A A A | 0
  52 | left   | (8, 1) | A A A | 0
  53 | drop   | (8, 1) | A A A | 0
  54 | right  | (8, 2) | A A A | 0
  55 | down   | (8, 2) | A A A | 0
  56 | up     | (7, 2) | A A A | 0
  57 | down   | (8, 2) | A A A | 0
  58 | noop   | (8, 2) | A A A | 0
  59 | down   | (8, 2) | A A A | 0
  60 | noop   | (8, 2) | A A A | 0
  61 | pick   | (8, 2) | A A A | 0
  62 | pick   | (8, 2) | A A A | 0
  63 | up     | (7, 2) | A A A | 0
  64 | drop   | (7, 2) | A A A | 0
  65 | down   | (8, 2) | A A A | 0
  66 | down   | (8, 2) | A A A | 0
  67 | left   | (8, 1) | A A A | 0
  68 | left   | (8, 1) | A A A | 0
  69 | right  | (8, 2) | A A A | 0
  70 | pick   | (8, 2) | A A A | 0
  71 | up     | (7, 2) | A A A | 0
  72 | drop   | (7, 2) | A A A | 0
  73 | drop   | (7, 2) | A A A | 0
  74 | down   | (8, 2) | A A A | 0
  75 | down   | (8, 2) | A A A | 0
  76 | noop   | (8, 2) | A A A | 0
  77 | drop   | (8, 2) | A A A | 0
  78 | up     | (7, 2) | A A A | 0
  79 | down   | (8, 2) | A A A | 0
  80 | drop   | (8, 2) | A A A | 0
  81 | pick   | (8, 2) | A A A | 0
  82 | drop   | (8, 2) | A A A | 0
  83 | drop   | (8, 2) | A A A | 0
  84 | pick   | (8, 2) | A A A | 0
  85 | down   | (8, 2) | A A A | 0
  86 | down   | (8, 2) | A A A | 0
  87 | up     | (7, 2) | A A A | 0
  88 | noop   | (7, 2) | A A A | 0
  89 | up     | (7, 2) | A A A | 0
  90 | noop   | (7, 2) | A A A | 0
  91 | up     | (7, 2) | A A A | 0
  92 | down   | (8, 2) | A A A | 0
  93 | up     | (7, 2) | A A A | 0
  94 | up     | (7, 2) | A A A | 0
  95 | drop   | (7, 2) | A A A | 0
  96 | pick   | (7, 2) | A A A | 0
  97 | drop   | (7, 2) | A A A | 0
  98 | drop   | (7, 2) | A A A | 0
  99 | noop   | (7, 2) | A A A | 0
 100 | pick   | (7, 2) | A A A | 0
 101 | noop   | (7, 2) | A A A | 0
 102 | down   | (8, 2) | A A A | 0
 103 | pick   | (8, 2) | A A A | 0
 104 | pick   | (8, 2) | A A A | 0
 105 | down   | (8, 2) | A A A | 0
 106 | drop   | (8, 2) | A A A | 0
 107 | drop   | (8, 2) | A A A | 0
 108 | noop   | (8, 2) | A A A | 0
 109 | left   | (8, 1) | A A A | 0
 110 | drop   | (8, 1) | A A A | 0
 111 | noop   | (8, 1) | A A A | 0
 112 | down   | (8, 1) | A A A | 0
 113 | left   | (8, 1) | A A A | 0
 114 | down   | (8, 1) | A A A | 0
 115 | pick   | (8, 1) | A A A | 0
 116 | right  | (8, 2) | A A A | 0
 117 | drop   | (8, 2) | A A A | 0
 118 | left   | (8, 1) | A A A | 0
 119 | down   | (8, 1) | A A A | 0
 120 | left   | (8, 1) | A A A | 0
 121 | down   | (8, 1) | A A A | 0
 122 | left   | (8, 1) | A A A | 0
 123 | right  | (8, 2) | A A A | 0
 124 | down   | (8, 2) | A A A | 0
 125 | drop   | (8, 2) | A A A | 0
 126 | left   | (8, 1) | A A A | 0
 127 | left   | (8, 1) | A A A | 0
 128 | right  | (8, 2) | A A A | 0
 129 | up     | (7, 2) | A A A | 0
 130 | down   | (8, 2) | A A A | 0
 131 | drop   | (8, 2) | A A A | 0
 132 | left   | (8, 1) | A A A | 0
 133 | drop   | (8, 1) | A A A | 0
 134 | left   | (8, 1) | A A A | 0
 135 | drop   | (8, 1) | A A A | 0
 136 | right  | (8, 2) | A A A | 0
 137 | pick   | (8, 2) | A A A | 0
 138 | left   | (8, 1) | A A A | 0
 139 | down   | (8, 1) | A A A | 0
 140 | right  | (8, 2) | A A A | 0
 141 | up     | (7, 2) | A A A | 0
 142 | left   | (7, 1) | A A A | 0
 143 | noop   | (7, 1) | A A A | 0
 144 | left   | (7, 1) | A A A | 0
 145 | up     | (6, 1) | A A A | 0
 146 | noop   | (6, 1) | A A A | 0
 147 | down   | (7, 1) | A A A | 0
 148 | noop   | (7, 1) | A A A | 0
 149 | down   | (8, 1) | A A A | 0
 150 | noop   | (8, 1) | A A A | 0
 151 | down   | (8, 1) | A A A | 0
 152 | right  | (8, 2) | A A A | 0
 153 | drop   | (8, 2) | A A A | 0
 154 | drop   | (8, 2) | A A A | 0
 155 | noop   | (8, 2) | A A A | 0
 156 | drop   | (8, 2) | A A A | 0
 157 | pick   | (8, 2) | A A A | 0
 158 | pick   | (8, 2) | A A A | 0
 159 | noop   | (8, 2) | A A A | 0
 160 | up     | (7, 2) | A A A | 0
 161 | noop   | (7, 2) | A A A | 0
 162 | up     | (7, 2) | A A A | 0
 163 | noop   | (7, 2) | A A A | 0
 164 | up     | (7, 2) | A A A | 0
 165 | up     | (7, 2) | A A A | 0
 166 | drop   | (7, 2) | A A A | 0
 167 | drop   | (7, 2) | A A A | 0
 168 | down   | (8, 2) | A A A | 0
 169 | pick   | (8, 2) | A A A | 0
 170 | up     | (7, 2) | A A A | 0
 171 | left   | (7, 1) | A A A | 0
 172 | left   | (7, 1) | A A A | 0
 173 | down   | (8, 1) | A A A | 0
 174 | down   | (8, 1) | A A A | 0
 175 | noop   | (8, 1) | A A A | 0
 176 | drop   | (8, 1) | A A A | 0
 177 | right  | (8, 2) | A A A | 0
 178 | pick   | (8, 2) | A A A | 0
 179 | pick   | (8, 2) | A A A | 0
 180 | pick   | (8, 2) | A A A | 0
 181 | down   | (8, 2) | A A A | 0
 182 | drop   | (8, 2) | A A A | 0
 183 | noop   | (8, 2) | A A A | 0
 184 | drop   | (8, 2) | A A A | 0
 185 | down   | (8, 2) | A A A | 0
 186 | pick   | (8, 2) | A A A | 0
 187 | drop   | (8, 2) | A A A | 0
 188 | left   | (8, 1) | A A A | 0
 189 | noop   | (8, 1) | A A A | 0
 190 | pick   | (8, 1) | A A A | 0
 191 | left   | (8, 1) | A A A | 0
 192 | pick   | (8, 1) | A A A | 0
 193 | left   | (8, 1) | A A A | 0
 194 | up     | (7, 1) | A A A | 0
 195 | drop   | (7, 1) | A A A | 0
 196 | down   | (8, 1) | A A A | 0
 197 | left   | (8, 1) | A A A | 0
 198 | down   | (8, 1) | A A A | 0
 199 | right  | (8, 2) | A A A | 0
 200 | up     | (7, 2) | A A A | 0
