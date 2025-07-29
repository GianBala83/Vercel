import cv2
import numpy as np
import os
from PIL import Image
import pandas as pd
import os

# ----------------------------------------------------------------------------------
# ETAPA 1: FUNÇÃO PARA REDIMENSIONAR A IMAGEM
# ----------------------------------------------------------------------------------
def redimensionar_imagem(caminho_entrada, caminho_saida, largura, altura):
    """Redimensiona uma imagem para as dimensões especificadas."""
    try:
        img = Image.open(caminho_entrada)
        img_redimensionada = img.resize((largura, altura))
        img_redimensionada.save(caminho_saida)
        return True
    except FileNotFoundError:
        print(f"   ❌ Erro na Etapa 1: O arquivo de entrada não foi encontrado em '{caminho_entrada}'")
        return False
    except Exception as e:
        print(f"   ❌ Ocorreu um erro na Etapa 1: {e}")
        return False

# ----------------------------------------------------------------------------------
# ETAPA 2: FUNÇÃO PARA RECORTAR A IMAGEM COM BASE EM UM TEMPLATE
# ----------------------------------------------------------------------------------
def recortar_secao_com_template(caminho_imagem_completa, caminho_imagem_template, caminho_saida):
    """Encontra e recorta uma seção (template) em uma imagem maior."""
    if not os.path.exists(caminho_imagem_completa) or not os.path.exists(caminho_imagem_template):
        print("   ❌ Erro na Etapa 2: Arquivo de imagem completa ou de template não encontrado.")
        return False

    imagem_completa = cv2.imread(caminho_imagem_completa)
    template = cv2.imread(caminho_imagem_template)

    if imagem_completa is None or template is None:
        print("   ❌ Erro na Etapa 2 ao carregar as imagens.")
        return False

    h_template, w_template = template.shape[:2]
    resultado_match = cv2.matchTemplate(imagem_completa, template, cv2.TM_CCOEFF_NORMED)
    _, _, _, max_loc = cv2.minMaxLoc(resultado_match)
    
    top_left = max_loc
    bottom_right = (top_left[0] + w_template, top_left[1] + h_template)

    imagem_recortada = imagem_completa[top_left[1]:bottom_right[1], top_left[0]:bottom_right[0]]

    try:
        cv2.imwrite(caminho_saida, imagem_recortada)
        return True
    except Exception as e:
        print(f"   ❌ Erro na Etapa 2 ao salvar a imagem recortada: {e}")
        return False

# ----------------------------------------------------------------------------------
# ETAPA 3: FUNÇÃO PARA ANALISAR AS RESPOSTAS EM COORDENADAS ESPECÍFICAS
# ----------------------------------------------------------------------------------
def analisar_respostas_por_coordenadas(caminho_imagem, caminho_saida_debug):
    """Analisa as respostas em uma imagem de formulário e salva um resultado visual."""
    if not os.path.exists(caminho_imagem):
        print(f"   ❌ Erro na Etapa 3: Arquivo não encontrado em '{caminho_imagem}'")
        return None

    coordenadas_map = {
        "Questão 6":  {"SIM": (912, 34), "NÃO": (975, 34)},
        "Questão 7":  {"SIM": (912, 95), "NÃO": (975, 95)},
        "Questão 8":  {"SIM": (912, 162), "NÃO": (975, 162)},
        "Questão 9":  {"SIM": (912, 230), "NÃO": (975, 230)},
        "Questão 10": {"SIM": (912, 297), "NÃO": (975, 297)},
        "Questão 11": {"SIM": (912, 356), "NÃO": (975, 356)},
    }
    
    imagem_original = cv2.imread(caminho_imagem)
    if imagem_original is None:
        print(f"   ❌ Erro na Etapa 3: Não foi possível carregar a imagem em '{caminho_imagem}'")
        return None

    imagem_debug = imagem_original.copy()
    cinza = cv2.cvtColor(imagem_original, cv2.COLOR_BGR2GRAY)
    
    respostas = {}
    raio_amostra = 7
    limiar_marcacao = 180 

    for questao, coords in sorted(coordenadas_map.items()):
        (x_sim, y_sim) = coords["SIM"]
        (x_nao, y_nao) = coords["NÃO"]

        # Verificar se as coordenadas estão dentro dos limites da imagem
        if not (0 <= y_sim - raio_amostra < y_sim + raio_amostra <= cinza.shape[0] and
                0 <= x_sim - raio_amostra < x_sim + raio_amostra <= cinza.shape[1]):
            respostas[questao] = "Coordenadas fora dos limites (SIM)"
            cv2.putText(imagem_debug, "X", (x_sim, y_sim), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)
            continue

        if not (0 <= y_nao - raio_amostra < y_nao + raio_amostra <= cinza.shape[0] and
                0 <= x_nao - raio_amostra < x_nao + raio_amostra <= cinza.shape[1]):
            respostas[questao] = "Coordenadas fora dos limites (NÃO)"
            cv2.putText(imagem_debug, "X", (x_nao, y_nao), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)
            continue


        roi_sim = cinza[y_sim - raio_amostra : y_sim + raio_amostra, x_sim - raio_amostra : x_sim + raio_amostra]
        intensidade_sim = np.mean(roi_sim)

        roi_nao = cinza[y_nao - raio_amostra : y_nao + raio_amostra, x_nao - raio_amostra : x_nao + raio_amostra]
        intensidade_nao = np.mean(roi_nao)

        resposta = "Não marcado"
        if intensidade_sim < limiar_marcacao and intensidade_sim < intensidade_nao:
            resposta = "SIM"
            cv2.rectangle(imagem_debug, (x_sim - 15, y_sim - 15), (x_sim + 15, y_sim + 15), (0, 255, 0), 2)
        elif intensidade_nao < limiar_marcacao and intensidade_nao < intensidade_sim:
            resposta = "NÃO"
            cv2.rectangle(imagem_debug, (x_nao - 15, y_nao - 15), (x_nao + 15, y_nao + 15), (0, 0, 255), 2)

        respostas[questao] = resposta

    try:
        cv2.imwrite(caminho_saida_debug, imagem_debug)
    except Exception as e:
        print(f"   ❌ Erro na Etapa 3 ao salvar a imagem de debug: {e}")
        return None
        
    return respostas

def processar_imagem_individual(caminho_entrada, caminho_template, caminho_saida):
    try:
        # Gerar nomes de arquivos temporários únicos para evitar colisões
        unique_id = os.path.splitext(os.path.basename(caminho_entrada))[0]
        caminho_redimensionado = os.path.join(os.path.dirname(caminho_entrada), f"{unique_id}_redim.jpeg")
        caminho_recortado = os.path.join(os.path.dirname(caminho_entrada), f"{unique_id}_recortado.jpeg")


        sucesso1 = redimensionar_imagem(caminho_entrada, caminho_redimensionado, 1124, 1600)
        if not sucesso1:
            return None

        sucesso2 = recortar_secao_com_template(caminho_redimensionado, caminho_template, caminho_recortado)
        if not sucesso2:
            os.remove(caminho_redimensionado) # Limpa o temp
            return None

        resultado = analisar_respostas_por_coordenadas(caminho_recortado, caminho_saida)

        # Limpar arquivos temporários
        if os.path.exists(caminho_redimensionado):
            os.remove(caminho_redimensionado)
        if os.path.exists(caminho_recortado):
            os.remove(caminho_recortado)

        return resultado
    except Exception as e:
        print(f"Erro no processamento individual: {e}")
        # Limpar quaisquer arquivos temporários que possam ter sido criados antes do erro
        temp_files_to_clean = [caminho_redimensionado, caminho_recortado]
        for f in temp_files_to_clean:
            if 'temp_' in f and os.path.exists(f): # Evita deletar arquivos que não são temporários
                os.remove(f)
        return None


# ----------------------------------------------------------------------------------
# BLOCO DE EXECUÇÃO PRINCIPAL (Não será usado pelo Flask, mas útil para testes diretos)
# ----------------------------------------------------------------------------------
if __name__ == "__main__":
    # --- CONFIGURAÇÕES ---
    pasta_entrada = "pasta_entrada"
    pasta_saida = "pasta_saida"
    arquivo_template = 'template.jpeg'
    arquivo_excel_final = "resultados_gerais.xlsx"
    
    # --- PREPARAÇÃO ---
    os.makedirs(pasta_saida, exist_ok=True)
    todos_os_resultados = []
    
    print(f"--- INICIANDO PROCESSAMENTO EM LOTE DA PASTA '{pasta_entrada}' ---")

    extensoes_validas = ['.jpg', '.jpeg', '.png', '.bmp', '.tiff']

    lista_arquivos = sorted(os.listdir(pasta_entrada))
    for nome_arquivo in lista_arquivos:
        extensao = os.path.splitext(nome_arquivo)[1].lower()
        if extensao not in extensoes_validas:
            continue

        print(f"\n▶️   Processando arquivo: {nome_arquivo}")

        caminho_completo_original = os.path.join(pasta_entrada, nome_arquivo)
        
        base_name = os.path.splitext(nome_arquivo)[0]
        # Use um prefixo para arquivos temporários para distingui-los dos de saída final
        caminho_redimensionado = os.path.join(pasta_saida, f"temp_redimensionado_{base_name}.jpeg")
        caminho_recortado = os.path.join(pasta_saida, f"temp_recortado_{base_name}.jpeg")
        caminho_analise_final = os.path.join(pasta_saida, f"analise_{base_name}.jpg")

        sucesso_etapa1 = redimensionar_imagem(
            caminho_entrada=caminho_completo_original,
            caminho_saida=caminho_redimensionado,
            largura=1124,
            altura=1600
        )
        if not sucesso_etapa1:
            print(f"   ❌ Falha ao redimensionar {nome_arquivo}. Pulando para o próximo.")
            continue

        sucesso_etapa2 = recortar_secao_com_template(
            caminho_imagem_completa=caminho_redimensionado,
            caminho_imagem_template=arquivo_template,
            caminho_saida=caminho_recortado
        )
        if not sucesso_etapa2:
            print(f"   ❌ Falha ao recortar {nome_arquivo}. Pulando para o próximo.")
            os.remove(caminho_redimensionado)
            continue

        resultados_finais = analisar_respostas_por_coordenadas(
            caminho_imagem=caminho_recortado,
            caminho_saida_debug=caminho_analise_final
        )
        
        # Limpar arquivos temporários
        if os.path.exists(caminho_redimensionado):
            os.remove(caminho_redimensionado)
        if os.path.exists(caminho_recortado):
            os.remove(caminho_recortado)
        
        if resultados_finais:
            print(f"   ✅ Análise de '{nome_arquivo}' concluída com sucesso.")
            resultados_finais['Arquivo'] = nome_arquivo
            todos_os_resultados.append(resultados_finais)
        else:
            print(f"   ❌ Falha ao analisar {nome_arquivo}.")

    # --- SALVAR EM EXCEL ---
    if todos_os_resultados:
        print(f"\n--- COMPILANDO RESULTADOS EM '{arquivo_excel_final}' ---")
        
        df = pd.DataFrame(todos_os_resultados)
        
        colunas_questoes = sorted(
            [col for col in df.columns if col != 'Arquivo'], 
            key=lambda col: int(col.split(' ')[-1])
        )
        
        # Reordena o DataFrame com a ordem correta
        df = df[['Arquivo'] + colunas_questoes]
        
        try:
            df.to_excel(arquivo_excel_final, index=False)
            print(f"✅ Planilha de resultados salva com sucesso em '{arquivo_excel_final}'!")
        except Exception as e:
            print(f"❌ Erro ao salvar o arquivo Excel: {e}")
            
    else:
        print("\nNenhum arquivo foi processado com sucesso. A planilha Excel não foi gerada.")

    print("\n--- PROCESSAMENTO EM LOTE CONCLUÍDO ---")