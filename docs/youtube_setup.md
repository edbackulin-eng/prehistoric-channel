# YouTube для німецького каналу: покроково

Поки цього немає, конвеєр не зупиняється: готові епізоди лежать у буфері
(артефакт `release-<slug>` рендера, 90 днів, запис у `state/episodes.json`).
Щойно з'явиться секрет `YT_TOKEN_JSON`, watchdog за ~30 хв завантажить їх
на найближчі вільні слоти (середа і субота, 17:00 за Берліном).

## 1. Канал
1. Увійди в YouTube тим Google-акаунтом, яким керуєш каналами.
2. Аватар → **Налаштування** → **Створити канал** (або «Додати чи
   керувати каналами» → **Створити канал**) — це канал бренду. Назва
   німецькою, наприклад «Urzeit-Archiv».
3. YouTube Studio → **Налаштування → Канал → Основна інформація**: країна
   Німеччина; **Налаштування за замовчуванням → Мова назви й опису** —
   німецька.
4. **Обов'язково:** https://www.youtube.com/verify — підтвердження телефоном.
   Без нього YouTube не дає завантажувати відео довші за 15 хв і ставити свої
   обкладинки, а наші епізоди тривають ~25 хв.

## 2. OAuth-клієнт (Google Cloud)
1. https://console.cloud.google.com → новий проєкт `prehistoric-channel`.
2. **APIs & Services → Library** → «YouTube Data API v3» → **Enable**.
3. **OAuth consent screen**: тип External; назва застосунку будь-яка; свій
   e-mail. Scopes можна не додавати. Потім **Publish app → In production**:
   у режимі «Testing» токен вмирає через 7 днів. Перевірка Google для
   власного використання не потрібна — при вході буде попередження
   «неперевірений застосунок», його можна пропустити («Advanced → Go to …»).
4. **Credentials → Create credentials → OAuth client ID → Desktop app**.
   Завантаж JSON і поклади як `C:\prehistoric-channel\client_secret.json`
   (`.gitignore` його не пускає в репозиторій).

## 3. Токен → Secrets
У терміналі на ПК:

```bash
cd /c/prehistoric-channel && pip install google-auth-oauthlib google-api-python-client
```

```bash
cd /c/prehistoric-channel && python -c "from pipeline import youtube; youtube.authorize('client_secret.json', 'youtube_token.json')"
```

Відкриється браузер: обери **саме німецький канал** (канал бренду), дозволь
обидва доступи (керування YouTube і завантаження відео).

```bash
cd /c/prehistoric-channel && gh secret set YT_TOKEN_JSON -R edbackulin-eng/prehistoric-channel < youtube_token.json
```

Значення ніде не друкується; `youtube_token.json` у `.gitignore`.

## 4. Перевірка
- GitHub → Actions → **watchdog** → Run workflow → `test_email=true`: на пошту
  має прийти тестовий лист (коментар з @-згадкою в issue «Run failures»).
- Протягом пів години після секрету: **upload** для першого епізоду з буфера,
  у Studio — приватне відео з датою публікації, німецькі субтитри,
  обкладинка, позначка «змінений або синтетичний вміст».
