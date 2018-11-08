from __future__ import division
from __future__ import unicode_literals
# based on... https://chrisalbon.com/python/data_visualization/matplotlib_percentage_stacked_bar_plot/
from builtins import zip
from builtins import range
from past.utils import old_div
import pandas as pd
import matplotlib.pyplot as plt
#
# Ants1
# correct pose: 68.46% (#18468 frames)
# wrong pose: 0.04% (#10 frames)
# unknown pose: 31.50%
#
# Sowbug3
# correct pose: 80.14% (#18015 frames)
# wrong pose: 6.89% (#1548 frames)
# unknown pose: 12.98%
#
# Ants3
# correct pose: 89.95% (#40443 frames)
# wrong pose: 0.17% (#77 frames)
# unknown pose: 9.88%
#
#
# Zebrafish
# correct pose: 88.14% (#66089 frames)
# wrong pose: 0.16% (#117 frames)
# unknown pose: 11.70%


raw_data = {'first_name': ['F-Ants1', 'i-Ants1', 'F-Sow3', 'i-Sow3', 'F-Ants3', 'i-Ants3', 'F-Zebr', 'i-Zebr'],
        'correct': [68.46, 71.68, 80.14, 70.60, 89.95, 82.38, 88.14, 88.00],
        'wrong':   [00.04, 00.32, 06.89, 15.28, 00.17, 05.25, 00.16, 00.63],
        'unknown': [31.50, 28.01, 12.98, 14.12, 09.88, 12.37, 09.88, 11.36]}
df = pd.DataFrame(raw_data, columns = ['first_name', 'correct', 'wrong', 'unknown'])


# Create a figure with a single subplot
f, ax = plt.subplots(1, figsize=(10,5))

# Set bar width at 1
bar_width = 1

# positions of the left bar-boundaries
bar_l = list(range(len(df['correct'])))

from math import floor
# positions of the x-axis ticks (center of the bars as bar labels)
tick_pos = [i+(old_div(bar_width,2)) + 0.2*(floor(old_div(i,2))) for i in bar_l]

# Create the total score for each participant
totals = [i+j+k for i,j,k in zip(df['correct'], df['wrong'], df['unknown'])]

# Create the percentage of the total score the correct value for each participant was
pre_rel = [i / float(j) * 100 for  i,j in zip(df['correct'], totals)]

# Create the percentage of the total score the wrong value for each participant was
mid_rel = [i / float(j) * 100 for  i,j in zip(df['wrong'], totals)]

# Create the percentage of the total score the unknown value for each participant was
post_rel = [i / float(j) * 100 for  i,j in zip(df['unknown'], totals)]

# Create a bar chart in position bar_1
ax.bar(bar_l,
       # using pre_rel data
       pre_rel,
       # labeled
       label='Correct',
       # with alpha
       alpha=0.9,
       # with color
       color='#019600',
       # with bar width
       width=bar_width,
       # with border color
       edgecolor='white'
       )

# Create a bar chart in position bar_1
ax.bar(bar_l,
       # using mid_rel data
       mid_rel,
       # with pre_rel
       bottom=pre_rel,
       # labeled
       label='Wrong',
       # with alpha
       alpha=0.9,
       # with color
       color='#ee3300',
       # with bar width
       width=bar_width,
       # with border color
       edgecolor='white'
       )

# Create a bar chart in position bar_1
ax.bar(bar_l,
       # using post_rel data
       post_rel,
       # with pre_rel and mid_rel on bottom
       bottom=[i+j for i,j in zip(pre_rel, mid_rel)],
       # labeled
       label='Unknown',
       # with alpha
       alpha=0.9,
       # with color
       color='#333333',
       # with bar width
       width=bar_width,
       # with border color
       edgecolor='white'
       )

# Set the ticks to be first names
plt.xticks(tick_pos, df['first_name'])
ax.set_ylabel("Percentage")
ax.set_xlabel("")

# Let the borders of the graphic
plt.xlim([min(tick_pos)-bar_width, max(tick_pos)+bar_width])
plt.ylim(-10, 110)

# rotate axis labels
plt.setp(plt.gca().get_xticklabels(), rotation=45, horizontalalignment='right')

plt.legend()
# shot plot
plt.show()