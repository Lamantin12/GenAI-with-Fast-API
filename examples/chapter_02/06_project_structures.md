```
flat-project
├── app
│   ├── services.py
│   ├── database.py
│   ├── models.py
│   ├── routers.py
│   └── main.py
├── requirements.txt
├── .env
├── .gitignore
```


```
nested-project
├── app
│    ├── main.py
│    ├── dependencies.py
│    └── services
│    │   ├── users.py
│    │   └── profiles.py
│    └── models
│    │   ├── users.py
│    │   └── profiles.py
│    └── routers
│        ├── users.py
│        └── profiles.py
├── requirements.txt
├── .env
├── .gitignore
```



```
modular-project
├── app
│   └── modules
│   │   ├── auth
│   │   │   ├── routers.py
│   │   │   ├── models.py
│   │   │   ├── dependencies.py
│   │   │   ├── guards.py
│   │   │   ├── services.py
│   │   └── users
│   │   │   ├── router.py
│   │   │   ├── models.py
│   │   │   ├── dependencies.py
│   │   │   ├── services.py
│   │   │   ├── mappers.py
│   │   │   ├── pipes.py
│   │   └── profiles
│   └── routers
│   │   └── users.py
│   └── providers
│   │   └── email.py
│   │   └── stripe.py
............
│   ├── settings.py  # global configs
│   ├── middlewares.py  # global middlewares
│   ├── models.py  # global models
│   ├── exceptions.py  # global exceptions
│   └── main.py
├── requirements.txt
├── .env
├── .gitignore
```
