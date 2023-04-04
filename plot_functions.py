from import_statements import *

def plot_training(history_dict, model_name, dataset_name):
    fig = go.Figure()
    fig.add_trace(go.Scatter(x=[x for x in range(NUM_EPOCHS)],
                             y=history_dict.history['accuracy'],
                             mode='lines',
                             name='Training',
                             ))
    fig.update_layout(title=model_name, xaxis_title='Epoch', yaxis_title='Accuracy')
    fig.write_html(f"history_plot_{model_name}_{dataset_name}.html")
