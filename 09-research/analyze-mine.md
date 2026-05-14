### Własna analiza

Cel pracy: rozpoznawanie wielu typów modulacji sygnałów jednocześnie w odebranym sygnale radiowym (multiple modulation signal recognition)

Strengths:
1. Dobrze wyjaśnione terminy naukowe oraz wprowadzenie do problemu, wraz z praktycznymi powodami i problemami.
2. Wykorzystanie algorytmu genetycznego do optymalizacji architektury sieci neuronowej (chociaż to bardziej jako ciekawostka, bo jest to wykorzystane w dość ograniczonym zakresie)
3. (
Praca została napisana w 2019 roku. W ostatnich latach rozpoczął się proces wdrażania AI do sieci telekomunikacyjnych. Przykładowo, sieć 6G jest już nazywana AI-native, gdzie rozwiązania AI są już częscią samej architektury sieci. Ciekawe wykorzystanie przed boomem.
)

Weaknesses:
1. Tematyka pracy jest ciekawa - wykorzystanie uczenia głębokiego w problemach natury telekomunikacyjnej, jednak wydaje się, że rezultat prac nie jest spektakularny - to zwykła sieć MLP, której tylko dwie warstwy są optymalizowane algorytmem genetycznym.
2. Praca tłumaczy i wyjaśnia dość podstawowe terminy, jak aktywacja w sieci neuronowej. Struktura pracy i język wskazują, że praca nadaje się bardziej na inżynierską lub magisterską. Używają wyłącznie funkcji sigmoidalnej, jako aktywacji. Nie stosują już powszechnie wykorzystywanych ReLU, czy technik regularyzacji, jak Droput itd.

Pytania do autorów:
1. Dlaczego używaja MSE z regularyzacją L2 w klasyfikacji? Cross-entropy jest standardem dla klasyfikacji. MSE powoduje zanikający gradient, a autorzy sami wskazują, że wykorzystują sieć głęboką. Co więcej, stosują sigmoidalną funkcję aktywacji.

Dodatkowy opis:
Praca posiada strukturę IMRAD.

### Neutral prompt run
https://claude.ai/share/ca092a7e-ddf9-438c-aa98-7da09dc77e4f
Neutralna ocena pracy przez model językowi wynosi 3/10.

### Aggressive prompt run
https://claude.ai/share/694fe162-57fe-4a03-bf36-5f70da0a84a7
Agresywna ocena pracy przez model językowi wynosi 2/10, przy czym model podkreśla, że ocenę 1/10 daje pracom, które zawierają błędy matematyczne lub sfabrykowane dane.