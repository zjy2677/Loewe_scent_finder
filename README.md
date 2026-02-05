# Loewe_scent_finder
This repo is a RAG-based chatbot for Loewe's scent finder. The databse for perfumes is a merged file from fragrantica and loewe.
## How to use this in Google Colab
Open colab notebook and open a cell, paste this command and run two times to make sure all google ai related packages are eliminated.
```bash
!pip uninstall -y google-generativeai google-ai-generativelanguage langchain langchain-core langchain-community langchain-google-genai langgraph langgraph-prebuilt
```

Then paste this short code to make sure you have elinated all the packages installed before.
```bash
!pip uninstall -y loewe-scent-finder
```

Next paste this code to install the package, this process should take around 15s - 25s.
```bash
!pip install "git+https://github.com/zjy2677/Loewe_scent_finder.git"
```

Finally, import the scent finder chatbot from downloaded package.
```bash
from Loewe_scent_finder_chatbot.main import main
```

Now run the main function to start chat
```bash
main()
```
## Strcuture of the repo 
