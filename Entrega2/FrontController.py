import dash
from dash import dcc, html, Input, Output, State, dash_table
import dash_bootstrap_components as dbc
import base64
import io
import pandas as pd

# Initialize the app
app = dash.Dash(__name__, external_stylesheets=[dbc.themes.BOOTSTRAP])

# Define the app layout
app.layout = dbc.Container(
    [
        dbc.Row(
            dbc.Col(
                html.H1("File Upload Example", className="text-center my-4")
            )
        ),
        dbc.Row(
            dbc.Col(
                dcc.Upload(
                    id='upload-data',
                    children=html.Div([
                        'Drag and Drop or ',
                        html.A('Select Files')
                    ]),
                    style={
                        'width': '100%',
                        'height': '60px',
                        'lineHeight': '60px',
                        'borderWidth': '1px',
                        'borderStyle': 'dashed',
                        'borderRadius': '5px',
                        'textAlign': 'center',
                        'margin': '10px'
                    },
                    multiple=False
                )
            )
        ),
        dbc.Row(
            dbc.Col(
                html.Div(id='output-data-upload')
            )
        )
    ],
    fluid=True
)

@app.callback(
    Output('output-data-upload', 'children'),
    Input('upload-data', 'contents'),
    State('upload-data', 'filename'),
)
def update_output(content, filename):    
    df = None

    print(content)
    print(filename)
    if content is not None:
        content_type, content_string = content.split(',')

        decoded = base64.b64decode(content_string)
        
        if filename.endswith('.csv'):
            df = pd.read_csv(decoded.decode('utf-8'))
        elif filename.endswith('.xlsx'):
            df = pd.read_excel(decoded.decode('utf-8'))
        else:
            return html.Div([
                'File type not supported.'
            ])
        # except Exception as e:
        #     return html.Div([
        #         'There was an error processing the file.'
        #     ])
        
        return html.Div([
            html.H5(f"Uploaded file: {filename}"),
            html.Hr(),
            dash_table.DataTable(data=df.to_dict('records'), page_size=10)
        ])
    return html.Div([
        "No file uploaded yet."
    ])

# Run the app
if __name__ == '__main__':
    app.run_server(debug=True)