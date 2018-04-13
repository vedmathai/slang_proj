echo 'This file will run through all the training and running of the three methods'
echo ''
echo 'SpaCy'
python3 train.py --engine spacy --out models/all_spacy.pkl trainfiles/all.json
echo
echo 'Testing sentence: "book me a table for four"'
python3 infer.py --engine spacy --model models/all_spacy.pkl 'book me a table for four'
echo
echo 'Testing sentence: "what is the weather like in chennai today"'
python3 infer.py --engine spacy --model models/all_spacy.pkl 'what is the weather like in chennai today'
echo
echo 'Testing sentence: "where is king kong playing today"'
python3 infer.py --engine spacy --model models/all_spacy.pkl 'where is king kong playing today'
echo
echo
echo 'Magic'
python3 train.py --engine magic --out models/all_magic.pkl trainfiles/all.json
echo
echo 'Testing sentence: "book me a table for four"'
python3 infer.py --engine magic --model models/all_magic.pkl 'book me a table for four'
echo
echo 'Testing sentence: "what is the weather like in chennai today"'
python3 infer.py --engine magic --model models/all_magic.pkl 'what is the weather like in chennai today'
echo
echo 'Testing sentence: "where is king kong playing today"'
python3 infer.py --engine magic --model models/all_magic.pkl 'where is king kong playing today'
echo
echo
echo 'Spacy Slang'
python3 train.py --engine spacySlang --out models/spacy_slang.pkl trainfiles/TravelMate.json
echo 'Testing sentence: Show me delhi'
python3 infer.py --engine spacySlang --model models/spacy_slang.pkl 'Show me delhi'
echo
echo 'Testing sentence: show me london'
python3 infer.py --engine spacySlang --model models/spacy_slang.pkl 'show me london'
echo
echo 'Testing sentence: Book my tickets to Narnia'
python3 infer.py --engine spacySlang --model models/spacy_slang.pkl 'Book my tickets to Narnia'
echo
echo 'Magic Slang'
python3 train.py --engine magicSlang --out models/magic_slang.pkl trainfiles/TravelMate.json
echo 'Testing sentence: Show me Calcutta'
python3 infer.py --engine magicSlang --model models/magic_slang.pkl 'Show me Calcutta'
echo
echo 'Testing sentence: How do I go from Cawnpore to Madras'
python3 infer.py --engine magicSlang --model models/magic_slang.pkl 'How do I go from Cawnpore to Madras'
echo
echo 'Testing sentence: Book my tickets to Narnia'
python3 infer.py --engine magicSlang --model models/magic_slang.pkl 'Book my tickets to Narnia'
