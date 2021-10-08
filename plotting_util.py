"""Plotting Utils."""

import altair as alt
import numpy as np
import pandas as pd



def similarity_heatmaps(sim_of_sim, labels_dict, axis_title='', width=300, columns=2, min_step=1):
  
  plot_data = pd.DataFrame()
  for key in sim_of_sim:
    # Compute x^2 + y^2 across a 2D grid
    labels = labels_dict[key] or range(len(sim_of_sim[key]))
    x, y = np.meshgrid(labels, labels)
    z = sim_of_sim[key]
    

    # Convert this grid to columnar data expected by Altair
    row = pd.DataFrame({'x': x.ravel(),
                        'y': y.ravel(),
                        'z': sim_of_sim[key].ravel(),
                        'key': key})
    plot_data = plot_data.append(row + text, ignore_index=True)

    
  base = alt.Chart(plot_data, width=width, height=width).mark_rect().encode(
      x=alt.X('x:N', sort=labels, title=axis_title, 
              axis=alt.Axis(values=np.asarray(labels)[list(range(0, len(labels), min_step))])),
      y=alt.X('y:N', sort=labels, title=axis_title, 
             axis=alt.Axis(values=np.asarray(labels)[list(range(0, len(labels), min_step))])),
      color=alt.Color('z:Q', title='Similarity'),
  )
  # Configure text
  text = base.mark_text(baseline='middle').encode(
        text='z:Q',
        color=alt.condition(
            alt.datum.z > 0.5,
            alt.value('black'),
            alt.value('white')
        )
    )
    
  plot = base + text

  plot.facet(
      facet=alt.Facet('key:N', title='', header=alt.Header(labelFontSize=16)),
      columns=columns
  ).resolve_scale(
      color='independent',
      x='independent',
      y='independent',
  ).configure_axis(
    labelFontSize=14,
    titleFontSize=16
).configure_legend(
    labelFontSize=14,
    titleFontSize=14
).configure_title(
    fontSize=18)



def layer_similarity(sim_of_sim, labels_dict, axis_title='', width=300, columns=2):
  
  plot_data = pd.DataFrame()
  for key in sim_of_sim:
    # Compute x^2 + y^2 across a 2D grid
    labels = labels_dict[key] or list(range(len(sim_of_sim[key])))
    z = sim_of_sim[key].diagonal(1)
    # Convert this grid to columnar data expected by Altair
    row = pd.DataFrame({'x': np.asarray(labels)[:-1],
                        'y': z,
                        'key': key})
    plot_data = plot_data.append(row, ignore_index=True)

  return alt.Chart(plot_data, width=width, height=width).mark_line(point=True).encode(
      x=alt.X('x:O', sort=labels, title='Layer', scale=alt.Scale(zero=False)),
      y=alt.X('y:Q', title='Sim(L(i), L(i+1))', scale=alt.Scale(zero=False)),
      color=alt.Color('key:N', title='Model', scale=alt.Scale(zero=False)),
  ).facet(
      facet=alt.Facet('key:N', title='', header=alt.Header(labelFontSize=16)),
      columns=columns
  ).resolve_scale(
      x='independent',
      y='independent',
  ).configure_axis(
    labelFontSize=14,
    titleFontSize=16
).configure_legend(
    labelFontSize=14,
    titleFontSize=14
).configure_title(
    fontSize=18)
