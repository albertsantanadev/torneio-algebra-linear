"""Gerador auxiliar de PDF para a copia de desenvolvimento (nao enviar ao avaliador)."""
from reportlab.platypus import SimpleDocTemplate, Paragraph, Spacer, Table, TableStyle, PageBreak
from reportlab.lib.styles import ParagraphStyle
from reportlab.lib import colors
from reportlab.lib.enums import TA_LEFT, TA_CENTER
from reportlab.pdfbase import pdfmetrics
from reportlab.pdfbase.ttfonts import TTFont
from reportlab.lib.pagesizes import A4

pdfmetrics.registerFont(TTFont('DVS', '/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf'))
pdfmetrics.registerFont(TTFont('DVS-B', '/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf'))

ARQUIVO = '/mnt/data/torneio_algebra_revisado/relatorio/relatorio_algoritmo.pdf'
AZUL = colors.HexColor('#152840')
AZULCLARO = colors.HexColor('#EAF1F9')
CINZA = colors.HexColor('#546579')
VERDE = colors.HexColor('#067C68')

styles = {
    'title': ParagraphStyle('titulo', fontName='DVS-B', fontSize=18, leading=23, textColor=AZUL, spaceAfter=9),
    'sub': ParagraphStyle('sub', fontName='DVS', fontSize=9, leading=15, textColor=CINZA, spaceAfter=16),
    'h': ParagraphStyle('secao', fontName='DVS-B', fontSize=11, leading=15, textColor=AZUL, spaceBefore=11, spaceAfter=6),
    'p': ParagraphStyle('corpo', fontName='DVS', fontSize=8.8, leading=14.2, textColor=colors.HexColor('#202C3B'), spaceAfter=9),
    'small': ParagraphStyle('pequeno', fontName='DVS', fontSize=8.15, leading=12.6, textColor=colors.HexColor('#283648'), spaceAfter=6),
    'th': ParagraphStyle('cabc', fontName='DVS-B', fontSize=8.3, leading=11, textColor=colors.white),
    'td': ParagraphStyle('cell', fontName='DVS', fontSize=8.2, leading=12, textColor=AZUL),
    'tdb': ParagraphStyle('cellbold', fontName='DVS-B', fontSize=8.2, leading=12, textColor=AZUL),
}

def P(s, style='p'):
    return Paragraph(s, styles[style])

def tab(dados, colWidths, header=True):
    out = []
    for i,row in enumerate(dados):
        out.append([P(str(c), 'th' if i == 0 and header else ('tdb' if i and str(c).startswith('MÉDIA') else 'td')) for c in row])
    t = Table(out, colWidths=colWidths, hAlign='LEFT', repeatRows=1 if header else 0)
    t.setStyle(TableStyle([
        ('BACKGROUND',(0,0),(-1,0),AZUL),
        ('ROWBACKGROUNDS',(0,1),(-1,-1),[colors.white, colors.HexColor('#F3F7FA')]),
        ('VALIGN',(0,0),(-1,-1),'TOP'),
        ('LEFTPADDING',(0,0),(-1,-1),10),('RIGHTPADDING',(0,0),(-1,-1),10),
        ('TOPPADDING',(0,0),(-1,-1),7),('BOTTOMPADDING',(0,0),(-1,-1),7),
        ('LINEBELOW',(0,-1),(-1,-1),0.4,colors.HexColor('#D6E0E8')),
    ]))
    return t

def page(canvas, doc):
    canvas.saveState()
    w,h=A4
    canvas.setFillColor(AZUL)
    canvas.rect(0,h-11,w,11,stroke=0,fill=1)
    canvas.setStrokeColor(colors.HexColor('#D6E0E8'))
    canvas.line(40,43,w-40,43)
    canvas.setFont('DVS',7.5)
    canvas.setFillColor(CINZA)
    canvas.drawString(41,29,'CIN0138 • Torneio de reconstrução de matrizes')
    canvas.drawRightString(w-41,29,f'17/11/2026 • Página {doc.page}')
    canvas.restoreState()

c = SimpleDocTemplate(ARQUIVO, pagesize=A4, leftMargin=43, rightMargin=43, topMargin=35, bottomMargin=57, title='Relatório do algoritmo — Torneio de Álgebra Linear', author='Equipe Albert, Erlon e Romildo')
W = A4[0] - 86
story = [
    P('Relatório do algoritmo', 'title'),
    P('Torneio de Álgebra Linear e Vetorial para Computação (CIN0138) &nbsp; | &nbsp; Equipe: Albert, Erlon e Romildo', 'sub'),
    P('1. Objetivo', 'h'),
    P('Implementar <b>principal(observada, mascara)</b> para completar matrizes de até <b>40 × 40</b> sem consultar entradas ocultas. A saída deve preservar as dimensões, manter os valores conhecidos e não conter NaN nem infinito. A classificação considera o NRMSE nas posições ocultas.'),
    P('2. Métodos considerados', 'h'),
    tab([
       ['Família', 'Ideia matemática'],
       ['Médias', 'Estimativa global, por linha ou por coluna.'],
       ['Aditivo regularizado', 'Média + efeito de linha + efeito de coluna; 18 iterações alternadas, com ridge de 0,5 ou 2,0.'],
       ['Linhas/colunas vizinhas', 'Interpolação entre os valores observados mais próximos ao longo de um eixo.'],
       ['Baixo posto (Hard-Impute)', 'SVD truncada com postos candidatos 1, 2, 3, 4, 5, 6 e 8 (quando permitidos); com e sem centralização.'],
    ], [W*0.29, W*0.71]),
    P('3. Escolha automática do algoritmo', 'h'),
    P('A matriz observada é separada em grupos determinísticos. Em cada teste, uma parte das <b>entradas conhecidas</b> é escondida temporariamente; cada candidato reconstrói a matriz usando somente as demais e é avaliado pelo erro quadrático médio nas células reservadas. Escolhemos o candidato de menor erro. São utilizadas três divisões para matrizes com até 600 células e duas para matrizes maiores, economizando processamento.'),
    P('No Hard-Impute, as posições faltantes são inicialmente preenchidas; a cada iteração calculamos uma SVD, mantemos as componentes associadas aos maiores valores singulares e reinserimos os valores efetivamente conhecidos. A reconstrução final utiliza até 85 iterações. Nenhum gabarito é utilizado pelo algoritmo.'),
    P('4. Segurança e casos especiais', 'h'),
    P('Se não houver valores conhecidos, devolvemos zeros. Para linhas ou colunas completamente ocultas, usamos a média global como fallback. Limitamos extrapolações excessivas, substituímos resultados não finitos pela média e restauramos exatamente os valores originalmente conhecidos. O arquivo de submissão utiliza somente <b>NumPy</b>, sem classes, aleatoriedade ou módulos auxiliares externos.'),
    PageBreak(),
    P('Resultados e validação', 'title'),
    P('Medições locais em 17/11/2026; resultados oficiais dependem do Docker atualizado.', 'sub'),
    P('5. Resultados no banco público', 'h'),
    P('Os dez gabaritos públicos foram utilizados somente para calcular métricas <b>após</b> executar as reconstruções. NRMSE = RMSE dividido por um desvio-padrão de referência. A monitoria ajustou a normalização da aplicação; por isso registramos separadamente o desvio das ocultas e o desvio da matriz completa.'),
    tab([
       ['Método', 'NRMSE (ocultas)', 'NRMSE (matriz total)'],
       ['Original: média global', '1,0471', '1,0215'],
       ['Protótipo anterior', '0,3275', '0,3323'],
       ['MÉDIA: versão integrada', '0,2668', '0,2728'],
    ], [W*0.48, W*0.26, W*0.26]),
    Spacer(1,10),
    tab([
       ['Matriz', 'Dimensões', 'NRMSE (ocultas)', 'Matriz', 'Dimensões', 'NRMSE (ocultas)'],
       ['01','8 × 8','1,2342', '06','16 × 16','0,0430'],
       ['02','9 × 6','0,0308', '07','18 × 22','0,0883'],
       ['03','6 × 9','0,0740', '08','17 × 17','1,0026'],
       ['04','14 × 5','0,0384', '09','24 × 19','0,0012'],
       ['05','14 × 7','0,1385', '10','28 × 31','0,0174'],
    ], [W*0.105, W*0.16, W*0.235, W*0.105, W*0.16, W*0.235]),
    P('6. Testes de robustez e tempo', 'h'),
    P('O verificador local aprovou as dez matrizes públicas, com <b>zero erros e zero avisos</b>. Em dois lotes de 31 matrizes sintéticas, incluindo 40 × 40, o NRMSE médio (desvio das ocultas) foi <b>0,6559</b> (semente local 7) e <b>0,6980</b> (semente local 31). Foram executados mais 71 casos extremos: nenhum falhou, e o maior tempo individual observado nesse conjunto foi <b>0,168 s</b>. O tempo pode variar conforme a máquina e o ambiente restrito.'),
    P('7. Limitações e próximos passos', 'h'),
    P('As matrizes 01 e 08 permanecem difíceis. Validação cruzada pode escolher um modelo inadequado quando existem poucos dados ou linhas/colunas totalmente ocultas. Os testes locais não comprovam o resultado no lote privado. <b>Antes da entrega, executar principal.py no Docker atualizado</b>, verificar os limites de 2 s por matriz e 25 s por lote e confirmar a normalização da aplicação.'),
    P('Reprodução: <b>python harness/verificador.py src/principal.py</b> e <b>python harness/testes_limites.py src/principal.py</b>. Entrega: principal.py e este relatório, ambos dentro de Grupo[Número].zip.', 'small'),
]
c.build(story,onFirstPage=page,onLaterPages=page)
print(ARQUIVO)
