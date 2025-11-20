# GitHubアップロード手順ガイド

このプロジェクトをGitHubにアップロードする手順です。

## ステップ1: Gitリポジトリの初期化

プロジェクトディレクトリで以下のコマンドを実行：

```powershell
git init
```

## ステップ2: ファイルをステージング

すべてのファイルを追加：

```powershell
git add .
```

特定のファイルのみ追加する場合：

```powershell
git add .gitignore
git add README.md
git add backend/
git add frontend/
```

## ステップ3: 初回コミット

```powershell
git commit -m "Initial commit: 寸法測定PoC実装"
```

## ステップ4: GitHubでリポジトリを作成

1. GitHubにログイン（https://github.com）
2. 右上の「+」ボタンをクリック → 「New repository」を選択
3. リポジトリ名を入力（例: `wall-size-measurement-poc`）
4. 説明を入力（任意）
5. **Public** または **Private** を選択
6. **「Initialize this repository with a README」はチェックしない**（既にREADMEがあるため）
7. 「Create repository」をクリック

## ステップ5: リモートリポジトリを追加

GitHubで作成したリポジトリのURLをコピーして、以下のコマンドを実行：

```powershell
git remote add origin https://github.com/あなたのユーザー名/リポジトリ名.git
```

例：
```powershell
git remote add origin https://github.com/username/wall-size-measurement-poc.git
```

## ステップ6: ブランチ名をmainに変更（必要に応じて）

```powershell
git branch -M main
```

## ステップ7: GitHubにプッシュ

```powershell
git push -u origin main
```

初回プッシュ時は、GitHubの認証情報を求められる場合があります。

## 完了！

これで、プロジェクトがGitHubにアップロードされました。

---

## トラブルシューティング

### 認証エラーが発生する場合

GitHubの認証方法を確認してください：
- Personal Access Token (PAT) を使用する
- GitHub CLI (`gh`) を使用する
- SSH キーを使用する

### 既存のリポジトリと競合する場合

```powershell
git pull origin main --allow-unrelated-histories
```

その後、再度プッシュ：

```powershell
git push -u origin main
```


