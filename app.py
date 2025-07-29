# app.py - Interface Flask para enviar imagens

from flask import Flask, render_template, request, jsonify, send_from_directory, Response
import os
import uuid
from processamento_imagens import processar_imagem_individual
import pandas as pd

app = Flask(__name__)
UPLOAD_FOLDER = 'uploads'
RESULT_FOLDER = 'resultados' # This will store the debug images and eventually the CSV
TEMPLATE_PATH = 'template.jpeg'  # caminho do template fixo no seu computador

os.makedirs(UPLOAD_FOLDER, exist_ok=True)
os.makedirs(RESULT_FOLDER, exist_ok=True)

# Configura o diretório para servir arquivos estáticos (CSS, JS)
app.static_folder = 'static'

@app.route('/')
def index():
    # Limpar resultados antigos ao carregar a página inicial
    # Esta é uma abordagem simples, em produção você pode querer uma estratégia mais robusta
    for folder in [UPLOAD_FOLDER, RESULT_FOLDER]:
        for filename in os.listdir(folder):
            file_path = os.path.join(folder, filename)
            try:
                if os.path.isfile(file_path) or os.path.islink(file_path):
                    os.unlink(file_path)
                elif os.path.isdir(file_path):
                    import shutil
                    shutil.rmtree(file_path)
            except Exception as e:
                print(f'Falha ao deletar {file_path}. Razão: {e}')
    return render_template('index.html')

@app.route('/upload', methods=['POST'])
def upload():
    if 'imagens' not in request.files:
        return jsonify({"success": False, "message": "Erro: Nenhum arquivo enviado."}), 400

    arquivos = request.files.getlist('imagens')
    if not arquivos:
        return jsonify({"success": False, "message": "Erro: Nenhum arquivo selecionado."}), 400

    todos_resultados = []
    debug_images_info = []

    for arquivo in arquivos:
        if arquivo.filename == '':
            continue

        original_filename = arquivo.filename
        
        # Gerar um nome único para o arquivo temporário e para o arquivo de análise
        # Usamos o nome original no resultado final para o usuário
        unique_id = str(uuid.uuid4())
        caminho_original = os.path.join(UPLOAD_FOLDER, f"{unique_id}_{original_filename}")
        caminho_analise = os.path.join(RESULT_FOLDER, f"analise_{unique_id}_{original_filename}.jpg")

        try:
            arquivo.save(caminho_original)

            resultados = processar_imagem_individual(
                caminho_entrada=caminho_original,
                caminho_template=TEMPLATE_PATH,
                caminho_saida=caminho_analise
            )

            # Remover o arquivo original após o processamento
            os.remove(caminho_original)

            if resultados:
                resultados['Arquivo'] = original_filename
                todos_resultados.append(resultados)
                debug_images_info.append({
                    "original_name": original_filename,
                    "debug_image_url": f"/resultados/{os.path.basename(caminho_analise)}"
                })
            else:
                # Se o processamento falhou, podemos registrar isso ou adicionar uma mensagem de erro
                print(f"Falha ao processar a imagem: {original_filename}")

        except Exception as e:
            print(f"Erro ao processar {original_filename}: {e}")
            # Tentar remover o arquivo original mesmo em caso de erro
            if os.path.exists(caminho_original):
                os.remove(caminho_original)
            # Adicionar uma entrada de erro para o frontend
            todos_resultados.append({'Arquivo': original_filename, 'Status': 'Erro no processamento'})

    if not todos_resultados:
        return jsonify({"success": False, "message": "Nenhuma imagem foi processada com sucesso."}), 500

    df = pd.DataFrame(todos_resultados)
    # Garante que 'Status' esteja no final se existir, ou que 'Arquivo' seja o primeiro
    cols = ['Arquivo'] + sorted([col for col in df.columns if col not in ['Arquivo', 'Status']], key=lambda col: int(col.split(' ')[-1]))
    if 'Status' in df.columns:
        cols.append('Status')
    df = df[cols]

    # Salvar resultados em CSV temporário para download
    csv_filename = f"resultados_analise_{uuid.uuid4()}.csv"
    csv_path = os.path.join(RESULT_FOLDER, csv_filename)
    df.to_csv(csv_path, index=False, encoding='utf-8-sig')

    return jsonify({
        "success": True,
        "results": df.to_dict(orient='records'),
        "debug_images": debug_images_info,
        "csv_download_url": f"/download_csv/{csv_filename}"
    })

@app.route('/resultados/<filename>')
def serve_resultado_image(filename):
    """Serve as imagens de debug da pasta RESULT_FOLDER."""
    return send_from_directory(RESULT_FOLDER, filename)

@app.route('/download_csv/<filename>')
def download_csv(filename):
    """Permite o download do arquivo CSV de resultados."""
    if not filename.endswith('.csv'):
        return "Arquivo inválido", 400
    try:
        return send_from_directory(RESULT_FOLDER, filename, as_attachment=True, mimetype='text/csv')
    except FileNotFoundError:
        return "Arquivo não encontrado", 404

if __name__ == '__main__':
    # Use 0.0.0.0 para que o servidor seja acessível externamente na rede local
    app.run(debug=True, host='0.0.0.0', port=5000)