# Goniometria Digital da Mao

> **Sistema de mensuracao goniometrica em tempo real para avaliacao funcional dos dedos da mao, utilizando visao computacional e interface desktop.**

[![Python](https://img.shields.io/badge/Python-3.11%2B-3776AB?style=flat-square&logo=python&logoColor=white)](https://www.python.org/)
[![PyQt6](https://img.shields.io/badge/PyQt6-6.6%2B-41CD52?style=flat-square&logo=qt&logoColor=white)](https://www.riverbankcomputing.com/software/pyqt/)
[![MediaPipe](https://img.shields.io/badge/MediaPipe-0.10%2B-0097A7?style=flat-square&logo=google&logoColor=white)](https://mediapipe.dev/)
[![OpenCV](https://img.shields.io/badge/OpenCV-4.9%2B-5C3EE8?style=flat-square&logo=opencv&logoColor=white)](https://opencv.org/)
[![License](https://img.shields.io/badge/Licenca-MIT-yellow?style=flat-square)](LICENSE)

---

## Sumario

- [Sobre o Projeto](#sobre-o-projeto)
- [Funcionalidades](#funcionalidades)
- [Arquitetura do Sistema](#arquitetura-do-sistema)
- [Tecnologias Utilizadas](#tecnologias-utilizadas)
- [Pre-requisitos](#pre-requisitos)
- [Instalacao](#instalacao)
- [Como Executar](#como-executar)
- [Estrutura do Projeto](#estrutura-do-projeto)
- [Configuracao](#configuracao)
- [Geracao de Relatorios](#geracao-de-relatorios)
- [Logs do Sistema](#logs-do-sistema)
- [Contribuindo](#contribuindo)

---

## Sobre o Projeto

O **Goniometria Digital da Mao** e uma aplicacao clinica de desktop desenvolvida para profissionais de saude — fisioterapeutas, terapeutas ocupacionais e medicos — que necessitam mensurar com precisao os angulos de flexao/extensao das articulacoes dos dedos da mao em tempo real.

O sistema utiliza a camera do computador, sem necessidade de qualquer equipamento fisico adicional (como o goniometro manual tradicional), e detecta automaticamente os pontos de referencia anatomicos (*landmarks*) da mao para calcular os angulos articulares das juntas **MCP**, **PIP**, **DIP** e, no polegar, **IP** e **ABD**.

### Contexto Clinico

A goniometria e o metodo padrao-ouro para avaliar a amplitude de movimento (ADM) articular. Em contexto de reabilitacao, medicoes frequentes e objetivas sao essenciais para acompanhar a evolucao do paciente. Este sistema digitaliza e acelera esse processo, eliminando o erro operador-dependente do goniometro fisico.

---

## Funcionalidades

- **Captura de video em tempo real** via webcam com resolucao HD (1280x720)
- **Deteccao automatica de landmarks** das 21 articulacoes da mao via MediaPipe
- **Calculo goniometrico em tempo real** para os 5 dedos (Indicador, Medio, Anelar, Minimo e Polegar)
- **Graficos dinamicos** com historico de angulos por articulacao (PyQtGraph)
- **Painel de metricas clinicas** com amplitude minima, maxima e media da sessao
- **Pipeline de suavizacao duplo** — Media Movel Exponencial (EMA) + Filtro de Kalman — para eliminar tremidos sem introduzir latencia
- **Gravacao de sessao em CSV** com timestamp e valores por articulacao
- **Geracao de relatorio em PDF** com sumario clinico da sessao
- **Interface Dark Mode** profissional e responsiva
- **Painel de logs em tempo real** embutido na interface
- **Configuracao centralizada** — todos os parametros em um unico arquivo `config.py`

---

## Arquitetura do Sistema

O sistema e construido sobre um padrao **Producer-Consumer com Workers Qt**, garantindo que a captura de video, o processamento de IA e a atualizacao da UI sejam completamente desacoplados e nao bloqueiem a interface grafica.

```
+------------------------------------------------------------------+
|                        app_pyqt.py                               |
|                  (Ponto de Entrada + Logging)                    |
+-------------------------+----------------------------------------+
                          |
                          v
+------------------------------------------------------------------+
|                    ui/main_window.py                             |
|              (Orquestrador Principal da UI)                      |
|                                                                  |
|  +--------------+  +--------------+  +---------------------+    |
|  | video_widget |  | plot_widget  |  | finger_card_widget  |    |
|  |  (Preview)   |  |  (Graficos)  |  |  (Cards por Dedo)   |    |
|  +--------------+  +--------------+  +---------------------+    |
|  +--------------+  +--------------+  +---------------------+    |
|  |session_header|  |metrics_widget|  |    log_widget       |    |
|  |  (Cabecalho) |  |  (Metricas)  |  | (Log tempo real)    |    |
|  +--------------+  +--------------+  +---------------------+    |
+---------------------------+--------------------------------------+
                            | Sinais Qt (thread-safe)
             +--------------+--------------+
             v                             v
+--------------------+         +----------------------+
|   workers/         |  Queue  |   workers/           |
|   CameraWorker     +-------->|   ProcessingWorker   |
|  (Thread Camera)   |  (=1)   |  (Thread IA/Calculo) |
+--------------------+         +----------+-----------+
                                          |
                              +-----+-----+------+
                              v     v            v
                         goniometry  smoothing  clinical_
                            .py        .py      classification.py
```

**Principios de design:**
- **Thread Safety**: Toda comunicacao entre threads usa sinais/slots Qt — nunca acesso direto a UI de threads secundarias.
- **Queue Size = 1**: A fila entre CameraWorker e ProcessingWorker tem tamanho maximo 1, garantindo que o processador sempre receba o frame mais recente (sem acumulo de latencia).
- **Configuracao centralizada**: Nenhum "numero magico" espalhado pelo codigo — tudo em `config.py`.

---

## Tecnologias Utilizadas

| Categoria | Tecnologia | Versao |
|-----------|-----------|--------|
| **Linguagem** | Python | 3.11+ |
| **Interface Grafica** | PyQt6 | 6.6+ |
| **Graficos em Tempo Real** | PyQtGraph | 0.13+ |
| **Visao Computacional** | MediaPipe | 0.10.11-0.10.17 |
| **Captura de Video** | OpenCV (cv2) | 4.9+ |
| **Calculos Matematicos** | NumPy | 1.24-1.x |
| **Relatorios PDF** | FPDF2 | 2.7+ |
| **Visualizacao de Dados** | Matplotlib | 3.7+ |
| **Monitoramento** | Logging (stdlib) | nativo |
| **Monitoramento de Recursos** | psutil | 5.9+ |

### Resumo Tecnico

* **Linguagem**: Python 3.11+
* **Interface**: PyQt6
* **Visao Computacional**: MediaPipe & OpenCV
* **Calculos Matematicos**: NumPy
* **Relatorios**: FPDF2
* **Logs do Sistema**: Modulo `logging` nativo

---

## Pre-requisitos

- **Sistema Operacional**: Windows 10/11 (recomendado), Linux ou macOS
- **Python**: 3.10 ou 3.11 (obrigatorio — limitacao do MediaPipe)
- **Camera**: Webcam integrada ou USB com resolucao minima de 720p
- **RAM**: Minimo 4 GB (recomendado 8 GB)
- **GPU**: Nao obrigatoria — o processamento e realizado em CPU

---

## Instalacao

### 1. Clone o repositorio

```bash
git clone https://github.com/seu-usuario/goniometria_digital.git
cd goniometria_digital
```

### 2. Crie e ative um ambiente virtual

```bash
# Windows
python -m venv .venv
.venv\Scripts\activate

# Linux/macOS
python3 -m venv .venv
source .venv/bin/activate
```

### 3. Instale as dependencias

```bash
pip install -r requirements.txt
```

> **Atencao (Windows):** Se ocorrer erro de SSL ao instalar o `aiortc`, execute:
> ```bash
> pip install aiortc==1.9.0
> ```

---

## Como Executar

### Interface Desktop (PyQt6) — Recomendado

```bash
python app_pyqt.py
```

---

## Estrutura do Projeto

```
goniometria_digital/
|
+-- app_pyqt.py                  # Ponto de entrada principal (PyQt6)
+-- config.py                    # Configuracao centralizada (parametros globais)
+-- goniometry.py                # Motor de calculo goniometrico (angulos articulares)
+-- goniometry_overlay.py        # Renderizacao do overlay na imagem da camera
+-- goniometry_csv.py            # Gravacao dos dados de sessao em CSV
+-- session_report.py            # Geracao do relatorio clinico em PDF
+-- smoothing.py                 # Pipeline de suavizacao (EMA + Filtro de Kalman)
+-- clinical_classification.py   # Classificacao clinica dos angulos medidos
+-- dashboard_utils.py           # Utilitarios para o painel de controle
+-- themes.py                    # Tema visual Dark Mode (estilos Qt)
+-- requirements.txt             # Dependencias do projeto
|
+-- ui/                          # Interface grafica (widgets PyQt6)
|   +-- main_window.py           #   Janela principal (orquestradora)
|   +-- video_widget.py          #   Widget de preview da camera
|   +-- plot_widget.py           #   Widget de graficos em tempo real
|   +-- finger_card_widget.py    #   Cards individuais por dedo
|   +-- metrics_widget.py        #   Painel de metricas clinicas
|   +-- session_header.py        #   Cabecalho da sessao
|   +-- log_widget.py            #   Painel de logs embutido
|
+-- workers/                     # Threads de processamento (Producer-Consumer)
|   +-- camera_worker.py         #   Thread de captura de video
|   +-- processing_worker.py     #   Thread de processamento IA + calculos
|
+-- assets/                      # Recursos estaticos (icones, imagens)
+-- logs/                        # Logs gerados pela aplicacao
+-- tests/                       # Testes automatizados
```

---

## Configuracao

Todos os parametros do sistema estao centralizados em [`config.py`](config.py). Nao e necessario alterar nenhum outro arquivo para ajustar o comportamento do sistema.

### Parametros principais

| Parametro | Valor Padrao | Descricao |
|-----------|-------------|-----------|
| `CAMERA_INDEX` | `0` | Indice da camera (0 = padrao) |
| `CAMERA_WIDTH` | `1280` | Largura da captura em pixels |
| `CAMERA_HEIGHT` | `720` | Altura da captura em pixels |
| `TARGET_FPS` | `30` | Taxa de quadros alvo |
| `EMA_ALPHA` | `0.30` | Fator de suavizacao EMA (0-1) |
| `KALMAN_Q` | `0.01` | Ruido do processo (Filtro de Kalman) |
| `KALMAN_R` | `0.10` | Ruido da medicao (Filtro de Kalman) |
| `MP_DETECT_CONF` | `0.70` | Confianca minima de deteccao (MediaPipe) |
| `MP_TRACK_CONF` | `0.50` | Confianca minima de rastreamento (MediaPipe) |
| `BUFFER_SIZE` | `500` | Pontos de historico nos graficos (~16s a 30 FPS) |
| `CSV_LOG_INTERVAL` | `3` | Frequencia de gravacao CSV (a cada N frames) |

### Trocar camera

Se o computador tiver multiplas cameras, altere em `config.py`:

```python
CAMERA_INDEX: int = 1  # 0 = padrao, 1 = camera externa, etc.
```

---

## Geracao de Relatorios

Ao encerrar uma sessao de avaliacao, o sistema gera automaticamente:

1. **Arquivo CSV** — contem os valores brutos de todos os angulos articulares com timestamp, gravados a aproximadamente 10 amostras/segundo.
2. **Relatorio PDF** — sumario clinico da sessao com amplitude minima, maxima e media por articulacao, gerado via [`session_report.py`](session_report.py) com a biblioteca FPDF2.

Os arquivos sao salvos na pasta raiz do projeto com o timestamp da sessao no nome do arquivo.

---

## Logs do Sistema

O sistema mantem dois niveis de registro:

| Tipo | Localizacao | Conteudo |
|------|------------|----------|
| **Log da Aplicacao** | `logs/app.log` | Eventos do sistema, erros, inicializacao |
| **Log de Sessao (CSV)** | Raiz do projeto | Dados clinicos (angulos por frame) |

O log da aplicacao utiliza o modulo nativo `logging` do Python, configurado em [`app_pyqt.py`](app_pyqt.py) para registrar simultaneamente no **console** (terminal) e no **arquivo** `logs/app.log`.

Formato padrao das mensagens:

```
2026-06-23 14:35:12,123 | INFO     | ui.main_window | Sessao iniciada
2026-06-23 14:35:45,891 | WARNING  | workers.camera | Frame descartado (fila cheia)
2026-06-23 14:36:02,045 | ERROR    | goniometry     | Landmarks insuficientes
```

---

## Contribuindo

Contribuicoes sao bem-vindas. Para contribuir:

1. Faca um fork do projeto
2. Crie uma branch para sua feature: `git checkout -b feature/minha-feature`
3. Commit suas mudancas: `git commit -m "feat: adiciona minha feature"`
4. Push para a branch: `git push origin feature/minha-feature`
5. Abra um Pull Request

### Padroes de codigo

- Siga as convencoes de nomenclatura existentes (`snake_case` para funcoes/variaveis, `UPPER_CASE` para constantes)
- Todo parametro numerico novo deve ser adicionado ao `config.py`, nunca inline no codigo
- Docstrings sao obrigatorias para funcoes e classes novas
- Mantenha os testes em `/tests` atualizados

---

## Licenca

Este projeto esta licenciado sob a licenca MIT. Consulte o arquivo [LICENSE](LICENSE) para mais detalhes.

---

Desenvolvido para aplicacao clinica em fisioterapia e terapia ocupacional.
