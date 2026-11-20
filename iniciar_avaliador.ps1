# Execute na pasta deste arquivo: .\iniciar_avaliador.ps1
# Garante que o SQLite seja um arquivo, nao uma pasta montada pelo Docker.
Set-Location -LiteralPath $PSScriptRoot
if (-not (Get-Command docker -ErrorAction SilentlyContinue)) {
    throw 'Docker nao encontrado. Abra o Docker Desktop antes de executar.'
}
$arquivo = Join-Path $PSScriptRoot 'competicao.db'
if (Test-Path -LiteralPath $arquivo -PathType Container) {
    throw 'competicao.db e uma PASTA. Pare o Docker, renomeie a pasta como backup e execute novamente.'
}
if (-not (Test-Path -LiteralPath $arquivo -PathType Leaf)) {
    New-Item -ItemType File -Path $arquivo | Out-Null
}
$historico = Join-Path $PSScriptRoot 'historico_local_tentativas'
if (-not (Test-Path -LiteralPath $historico -PathType Container)) {
    New-Item -ItemType Directory -Path $historico | Out-Null
}
docker compose pull
if ($LASTEXITCODE -ne 0) { throw 'Falha em docker compose pull' }
docker compose up -d
if ($LASTEXITCODE -ne 0) { throw 'Falha em docker compose up -d' }
docker compose ps
Write-Host 'Acesse http://localhost:8501'
