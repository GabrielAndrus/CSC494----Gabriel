# Prompts in english and local language for each location to generate the initial data 
generation_prompts={"England_gen":[
          {
                "role": "system",
                "content": "You are a cultural commonsense knowledge extraction assistant"
            },
            {
                "role": "user",
                "content": '''Generate as many if-then cultural commonsense knowledge statements as possible for {location} related to {sub_topic}. Each statement should follow this structure: "If [action], then [knowledge]." The knowledge should describe what action is likely to follow or precede the initial action, reflecting cultural habits and practices associated with {sub_topic} in {location}. 

        Let's think step by step to generate the chain of action between the action and the knowledge. The knowledge must be classified into one of the following relation types: 
            - xEffect: What effects does the action have on x? 
            - xNext: What would x likely want to do after the action? 
            - xNeed: What does x need to do before the action? 
            - oNext: What would others likely want to do after the action? 
            - oEffect: What effects does the action have on others?

        In the relation types above, 'x' represents the agent or primary person performing the action, while 'o' refers to others who interact with or are affected by the action of 'x'. 

        Format your response strictly as a JSON array of objects, following this precise format without any additional text or explanation:

        [
            {{
                "action": "action",
                "knowledge": "knowledge",
                "relation_type": "relation type",
                "result": "Complete sentence in {language}, using If and Then"
            }},
            ...
        ]

        Ensure that the "action," "knowledge," and the entire "result" sentences are generated in {language}, while the "relation_type" should be one of the specified options above.'''
            }
        ],

    "Indonesia_gen": [
          {
                "role": "system",
                "content":"Kamu adalah seorang pakar ilmu budaya yang sangat familiar dan paham tradisi dan nilai-nilai budaya di negara Anda, {location}"
            },
            {
                "role": "user",
                "content": '''Silakan tuliskan sebanyak mungkin pernyataan tentang pengetahuan budaya yang mencerminkan pemahaman umum (commonsense) dalam kehidupan sehari-hari di {location}, khususnya mengenai {sub_topic}, dengan menggunakan format hubungan sebab-akibat "jika-maka". Setiap pernyataan harus mengikuti struktur berikut: "Jika [action]; maka [knowledge]." Komponen knowledge harus mendeskripsikan aksi yang kemungkinan akan mengikuti atau mendahului action (tindakan) awal, serta mencerminkan kebiasaan dan tradisi budaya yang terkait dengan {sub_topic} di {location}.

           Mari kita berfikir secara terstruktur, langkah demi langkah untuk  untuk membangun deretan action baru yang berada di antara pasangan action dan knowledge yang telah ada. Penting untuk diingat bahwa setiap knowledge harus diklasifikasikan ke dalam salah satu jenis relasi berikut:

            - xEffect: Apa pengaruh action terhadap x (pelaku action)? 
            - xNext: Apa yang ingin dilakukan oleh x (pelaku action) setelah melakukan action? 
            - xNeed: Hal apa yang perlu dilakukan sebelum x (pelaku action) melakukan action tersebut?
            - oNext: Apa yang dilakukan oleh orang lain setelah x melakukan action tersebut?
            - oEffect: Apa pengaruh action terhadap orang lain?

        Dalam jenis relasi di atas, 'x' merupakan pelaku utama yang melakukan tindakan, sedangkan 'o' merujuk pada orang lain yang berinteraksi dengan 'x' atau dipengaruhi oleh 'x'.

        Gunakan objek array JSON sebagai respon Anda tanpa teks atau penjelasan tambahan, sebagai berikut:
        [
            {{
                "action": "action",
                "knowledge": "knowledge",
                "relation_type": "tipe relasi",
                "result": "Kalimat lengkap dalam {language}, menggunakan Jika dan Maka"
            }},
            ...
        ]

        Pastikan kalimat-kalimat "action," "knowledge," and "result" menggunakan {language}, sementara "relation_type" harus menggunakan salah satu tipe relasi yang telah ditentukan di atas.'''
            }
        ],
    "China_gen": [
    {
        "role": "system",
        "content":  "你是一名文化知识专家，你对你的国家（{location}）的传统和区域价值观非常熟悉。"
    },
    {
        "role": "user",
        "content": '''请尽量多生成于{location}的{sub_topic}相关的文化常识语句。 请使用“如果-那么”句式，每条语句应遵循以下结构：“如果[行为]，那么[知识]。” 该知识应描述在初始行为之后或之前可能采取的行为，且反映出与{location}的{sub_topic}相关的文化习惯和习俗。

            让我们一步步思考，从行为和知识之间生成一系列的动作链。这些知识必须分类为以下关系类型之一：
            - xEffect：该行动对x有哪些影响？
            - xNext：x在完成这一行动后可能想做什么？
            - xNeed：x在完成这一行动前需要做什么？
            - oNext：其他人在该行动后可能想做什么？
            - oEffect：该行动对其他人有什么影响？

            在上述关系类型中，“x”代表执行此行为的主体或主要人物，而“o”代表与“x”的行为相关或受到影响的其他人。

            严格按照以下JSON数组的格式生成您的回答，不要附加任何多余文本或解释：

[
{{
            "action": "行为",
            "knowledge": "知识",
            "relation_type": "关系",
            "result": "用{language}生成完整句子，以‘如果’和‘那么’开头\"
 }}, 
 ...
 ] 

确保“action”、“knowledge”和整个“result”语句用{language}生成，而“relation_type”应使用以上指定的选项之一。'''
    }
],
"Japan_gen": [
            {   
                "role": "system",
                "content": "あなたは文化知識の専門家であり、あなたの国である{location}の伝統や地域の価値観に深く精通したアシスタントです。"
            },
            {
                "role": "user",
                "content": '''{location}における{sub_topic}に関連するif-then形式の文化的な常識知識をできるだけたくさん生成してください。それぞれの生成結果は次の構造に従います：「もし [行動]したら、 [知識]。」 それぞれの知識は、その行動に対してどんな行動が先行しそうか、または続いて起こりそうかについて説明しており、{location}における{sub_topic}に関連する文化的慣習・慣例を反映する必要があります。"
            その行動と知識の間に想定される行動の連鎖について、段階的に生成してください。知識は以下の関係タイプのいずれかに分類されなければいけません： 
            - xEffect: その行動は、xにどのような影響を与えるか？ 
            - xNext: その行動の後、xは何をしたいと思うか？
            - xNeed: その行動の前に、xは何をする必要があるか? 
            - oNext: その行動の後に、他の人は何をしたいと思うか？
            - oEffect: その行動は、他者にどのような影響を与えるか？

        上記の関係タイプにおいて、"x"は行動の行為者または主たる動作主を、"o"は"x"の行動に関与する、または影響を受ける他者を指します。

        あなたの応答は厳密に以下のJSON配列形式に従い、以下のフォーマットに正確にならい、追加のテキストや説明は含めないでください：
        [
            {{
                "action": "行動",
                "knowledge": "知識",
                "relation_type": "関係タイプ",
                "result": "「もし...したら」を使った文を、{language}で生成してください"
            }},
            ...
        ]
        

        "行動"、"知識"、"結果"の文全体は、{language}で生成してください。ただし "relation_type" は、上記で制定された選択肢のうちのいずれかにしてください。'''
            }
        ], 
"Egypt_gen": [
    {
        "role": "system",
        "content": "أنت مساعد خبير في المعرفة الثقافية، على دراية عميقة بالتقاليد والقيم الإقليمية في بلدك، {location}"
    },
    {
        "role": "user",
        "content": '''قم بتوليد أكبر عدد ممكن من عبارات المعرفة الثقافية المرتبطة بـ"إذا-فإن" المتعلقة بـ{location} و{sub_topic}. يجب أن تتبع كل عبارة هذا الهيكل: "إذا [action], إذن [knowledge]." يجب أن تصف المعرفة ما هو الإجراء المحتمل الذي قد يتبع أو يسبق الإجراء الأول، مما يعكس العادات والممارسات الثقافية المرتبطة بـ{sub_topic} في {location}.

دعونا نفكر خطوة بخطوة لتوليد سلسلة العمل بين الإجراء والمعرفة. يجب تصنيف المعرفة إلى أحد أنواع العلاقات التالية: 
- xEffect: ما هي تأثيرات الإجراء على "x"؟
- xNext: ماذا سيود "x" القيام به على الأرجح بعد الإجراء؟
- xNeed: ماذا يحتاج "x" إلى فعله قبل الإجراء؟
- oNext: ماذا سيود الآخرون القيام به على الأرجح بعد الإجراء؟
- oEffect: ما هي تأثيرات الإجراء على الآخرين؟

في أنواع العلاقات أعلاه، تشير 'x' إلى الوكيل أو الشخص الأساسي الذي يقوم بالإجراء، بينما تشير 'o' إلى الآخرين الذين يتفاعلون مع أو يتأثرون بالإجراء الذي يقوم به 'x'.

قم بتنسيق استجابتك بدقة كصفيف JSON من الكائنات، باتباع هذا التنسيق الدقيق بدون أي نصوص أو شروح إضافية:

[
    {{
        "action": "action",
        "knowledge": "knowledge",
        "relation_type": "relation type",
        "result": "اكمل الجملة في {language}, بإستخدام إذا و إذن"
    }},
    ...
]

تأكد من أن جمل "action"، و"knowledge"، والجملة الكاملة في "result" مكتوبة بـ{language}، بينما يجب أن تكون "relation_type" واحدة من الخيارات المحددة أعلاه.'''
}
]
,
"Germany_gen": [
          {
                "role": "system",
                "content": "Sie sind ein Assistent, der Experte für kulturelles Wissen ist und mit den Traditionen und regionalen Werten Ihres Landes, {location}, bestens vertraut ist."
            },
            {
                "role": "user",
                "content": '''Erstellen Sie so viele kulturelle Wenn-Dann-Aussagen über kulturelles Allgemeinwissen wie möglich für {location} in Bezug auf {sub_topic}. Jede Aussage sollte folgende Struktur haben: "Wenn [Aktion], dann [Wissen]." Das Wissen sollte beschreiben, welche Handlung wahrscheinlich der ursprünglichen Handlung folgt oder vorausgeht und kulturelle Gewohnheiten und Praktiken im Zusammenhang mit {sub_topic} in {location} widerspiegeln. 

        Denken wir Schritt für Schritt darüber nach, um die Handlungskette zwischen der Aktion und dem Wissen zu generieren. Das Wissen muss in eine der folgenden Beziehungstypen klassifiziert werden: 
            - xEffect: Welche Auswirkungen hat die Aktion auf x? 
            - xNext: Was würde x nach der Aktion wahrscheinlich tun wollen? 
            - xNeed: Was muss x tun, bevor die Aktion stattfindet? 
            - oNext: Was würden andere nach der Aktion wahrscheinlich tun wollen? 
            - oEffect: Welche Auswirkungen hat die Aktion auf andere?

        In den obigen Beziehungstypen steht 'x' für die handelnde Person oder die Hauptperson, die die Aktion ausführt, während 'o' sich auf andere bezieht, die mit der Aktion von 'x' interagieren oder davon betroffen sind. 

        Formatieren Sie Ihre Antwort strikt als JSON-Array von Objekten, und folgen Sie dabei diesem präzisen Format ohne zusätzlichen Text oder Erklärungen:

        [
            {{
                "action": "Aktion",
                "knowledge": "Wissen",
                "relation_type": "Beziehungstyp",
                "result": "Vollständiger Satz auf {language}, unter Verwendung von Wenn und Dann"
            }},
            ...
        ]

        Stellen Sie sicher, dass die "action", "knowledge" und die gesamten "result"-Sätze in {language} generiert werden, während der "relation_type" einer der oben angegebenen Optionen entspricht.'''
            }
        ]
}


# Prompts in english and local language for each location to extend xNext/oNext relation

extension_prompts={"England_ext": [{ "role": "system", "content": "You are a cultural commonsense knowledge extraction assistant"
}, { "role": "user",  "content": '''Your task is to generate as many culturally sensitive commonsense knowledge events as possible for {location}, focusing on {sub_topic}, based on an initial event comprising an action and a knowledge. Each event must follow the structure: "If [action], then [knowledge]," and should accurately reflect the cultural habits, practices, traditions, and customs related to {sub_topic} in {location}.

Initial event: {initial_event}
initial action: {init_action}
initial knowledge: {init_knowledge}

Two types of generations are requested:
    1. Next steps generation
        - Start from the entire initial event 
        - use the initial knowledge as the starting point to generate all new knowledges
        - Ensure that each new knowledge logically derives from the initial event as a whole, taking into account both the initial action and the initial knowledge. 
        - Each new knowledge must logically align with the initial action and initial knowledge, showing their natural progression. Specifically, focus on the relationship types listed below.
        - Format the new knowledge as conditional statements, starting with: “if {init_knowledge}, then…”
        -  Each new knowledge must be classified into one of these relation types: 
              * xNext: What would x likely want to do after the action? 
              * oNext: What would others likely want to do after the action? 
            In the relation types above, 'x' represents the agent or primary person performing the action, while 'o' refers to others who interact with or are affected by the action of 'x'. 
    2. Intermediate steps generation
        - Start from the initial action from the initial event.
        - Create a chain of new knowledge (A_1, A_2, .., A_i) leading to the initial knowledge.
        - Use a stepwise approach:
            * Start with: if {init_action}, then A_1
            * then: “if A_1, then A_2”
            * generate new knowledge (A_i) that logically and culturally connects to the previous step. Continue iterating until you arrive at the final step, where the generated knowledge aligns with the initial knowledge.
            * Please respect the stepwise approach
        - Each step must be classified into one of the relation types listed above.
        - The chain should logically explain how someone would arrive at the initial knowledge in a culturally sensitive way.

Other requirements:
    1. Cultural relevance 
        Ensure all steps (new knowledge) are deeply rooted in and reflective of the culture in {location}, while being closely associated with the sub-topic {sub_topic} in {location}.
    2. Output format and output language
        Entire event sentence, action, knowlege in both next steps and intermediate steps must be written in {language} language.
        Your response must be formatted strictly as a JSON array of objects without any additional text or explanation and organized into two categories: next steps  and intermediate steps:
 
        [ {{"intermediate_steps": [
                {{
                    "action": " action",
                    "knowledge": "knowledge",
                    "relation_type": "relation",
                    "event": "If action, then knowledge"
                }}
                    ],
    "next_steps": [
                {{
                    "action": " action",
                    "knowledge": "knowledge",
                    "relation_type": "relation",
                    "event": "If action, then knowledge"
                }}
                 ]
            }}
        ]
'''
}],
"Indonesia_ext": [{ "role": "system", "content": "Kamu adalah seorang pakar ilmu budaya yang sangat familiar dan paham tradisi dan nilai-nilai budaya di negara Anda, {location}"
}, { "role": "user",  "content": ''' Anda diminta untuk menghasilkan sebanyak mungkin pengetahuan budaya yang mencerminkan pemahaman umum (commonsense) dalam kehidupan sehari-hari di {location}, dengan topik {sub_topic}. Data yang dimaksud merupakan event-event yang diekstrak dari event awal dengan struktur: "Jika [action], Maka [knowledge]". Setiap event harus mengikuti struktur tersebut dan harus mencerminkan tradisi dan adat istiadat yang terkait dengan topik {sub_topic} di {location}.

event awal: {initial_event}
action awal: {init_action}
knowledge awal: {init_knowledge}

Ada dua generasi teks yang diminta:
    1. Generasi Langkah Berikutnya (Next Steps Generation)
        - Mulailah dari koleksi event awal. 
        - Gunakan knowledge awal sebagai titik permulaan untuk membangkitkan knowledge-knowledge baru.
        - Pastikan setiap knowledge baru yang dihasilkan secara logis mengikuti event awal secara keseluruhan, mempertimbangkan baik action maupun knowledge awal.
        - Setiap knowledge baru harus secara logis selaras dengan action dan knowledge awal, dan menunjukkan perkembangan yang natural.
        - Knowledge baru merupakan pernyataan sebab-akibat, dimulai dengan: “Jika {init_knowledge}, Maka…”
        - Setiap knowledge baru hanya menggunakan tipe relasi berikut: 
              * xNext: Apa yang ingin dilakukan oleh x (pelaku action) setelah melakukan action? 
              * oNext: Apa yang dilakukan oleh orang lain setelah x melakukan action tersebut?
           Dalam jenis relasi di atas, 'x' merupakan pelaku utama yang melakukan tindakan, sedangkan 'o' merujuk pada orang lain yang berinteraksi dengan 'x' atau dipengaruhi oleh 'x'.
    2. Generasi Langkah Perantara (Intermediate Steps Generation)
        - Mulailah dari action awal yang merupakan komponen dari event awal.
        - Buatlah deretan knowledge baru (A_1, A_2, .., A_i) yang secara natural mengarah ke knowledge awal.
        - Gunakan pendekatan bertahap:
            * Mulai dengan: Jika {init_action}, Maka A_1
            * Kemudian: “Jika A_1, Maka A_2”
            * Bangkitkan knowledge baru (A_i) yang secara logis relevan dan terhubung ke event sebelumnya. Lanjutkan iterasi ini hingga kamu mencapai langkah akhir, initial knowledge.
            * Harap patuhi pendekatan bertahap ini.
        - Setiap langkah harus diklasifikasikan kedalam salah satu relasi yang telah dicantumkan di atas.
        - Deretan knowledge baru yang dihasilkan harus secara logis menjelaskan bagaimana seseorang bisa mencapai knowledge awal yang telah ditentukan sebelumnya.
 
Kriteria lain yang harus diikuti:
    1. Relevansi budaya 
        Pastikan semua knowledge-knowledge baru yang dihasilkan relevan dengan budaya yang ada di {location}, dan berkaitan dengan topik {sub_topic} di {location}.
    2. Format dan bahasa output
        Semua kalimat-kalimat yang ada pada event, action, dan knowledge di kedua "next steps" and "intermediate steps" harus ditulis dalam {language}.
        Respon Anda harus dalam objek array JSON tanpa penjelasan dan teks tambahan, dan distrukturkan dalam dua kategori: "next steps" dan "intermediate steps": 
        [{{"intermediate_steps": [
        {{
            "action": " action",
            "knowledge": "knowledge",
            "relation_type": "relation",
            "event": "Jika action, Maka knowledge"
        }}
    ],
    "next_steps": [
        {{
            "action": " action",
            "knowledge": "knowledge",
            "relation_type": "relation",
            "event": "Jika action, Maka knowledge"
        }}
    ]
}}]
'''
}],
"China_ext": [
  {
    "role": "system",
    "content": "你是一名文化知识专家，你对你的国家（{location}）的传统和区域价值观非常熟悉。"
  },
  {
    "role": "user",
    "content": '''你的任务是根据初始事件（包括一个行为和一个知识），为{location}尽可能多生成具有文化敏感的常识知识事件，请重点关注{sub_topic}。每个生成事件都应遵循以下结构：“如果[行为]，那么[知识]”。每个生成事件必须精准的反映与{location}的{sub_topic}相关的文化习惯、实践、传统和风俗。
初始事件：{initial_event}
初始行为：{init_action}
初始知识：{init_knowledge}
请生成两种类型的知识：
1. 后续步骤生成
   - 从初始事件开始。  
   - 使用初始知识作为起点生成所有新的知识。
   - 确保每个新知识在逻辑上都遵循整个初始事件，同时考虑初始行动和初始知识。
   - 每项新知识都必须与初始行动和初始知识在逻辑上保持一致，展现其自然发展。具体来说，重点关注以下列出的关系类型。
   - 将新知识格式化为条件语句，以“如果{init_knowledge}，那么..”开头 
   - 每个新知识必须归类为以下关系类型之一：
       - xNext：行动者 (x) 在行为之后可能想要做什么？ 
       - oNext：其他人 (o) 在行为之后可能想要做什么？
在上述关系类型中，“x”表示行动者, 执行动作的主体或主要人员，而“o”指与“x”的动作交互或受其影响的其他人。
2. 中间步骤生成
  - 从初始事件中的初始行为开始。
  - 创建一条通向初始知识的新知识链 （A_1, A_2, …, A_i）。
  - 采用逐步生成的方法：
     * 开始：“如果 {init_action}，那么 A_1”。
     * 然后：“如果 A_1，那么 A_2”。
     * 生成新知识（A_i），使其在逻辑和文化上与前一步紧密相连。继续迭代，直至最后一步生成的知识与初始知识一致。
     * 请遵循逐步生成方法。
  - 每个步骤都必须归类到上面列出的关系类型之一。
  - 这个链条应该以一种具有文化敏感性的方式，从逻辑上解释人们如何获得最初的知识。
其他要求:
1. 文化相关性
  确保所有步骤（新知识）深深植根于、并反映{location}的文化，同时与{location}的子主题：{sub_topic}紧密相关。
2. 输出格式与语言 
在后续步骤和中间步骤中，整个事件句子、行为和知识都必须用{language}语言书写。
你的回答必须严格遵循JSON数组对象的格式，不包含任何额外文本或解释。输出分为两类：后续步骤和中间步骤：
[{{"intermediate_steps": [
        {{
            "action": "行为",
            "knowledge": "知识",
            "relation_type": "关系",
            "event": "如果行为，那么知识"
        }}
    ],
    "next_steps": [
        {{
            "action": "行为",
            "knowledge": "知识",
            "relation_type": "关系",
            "event": "如果行为，那么知识"
        }}
    ]
}}]
'''
}],
"Japan_ext": [{ "role": "system", "content": "あなたは文化知識の専門家であり、あなたの国である{location}の伝統や地域の価値観に深く精通したアシスタントです。"
}, { "role": "user",  "content": '''あなたの任務は、{location}における{sub_topic}に関する文化に関連する常識的なイベント知識を可能な限り多く生成することです。このイベントは、1つの行動と1つの知識から構成される初期イベントに基づいています。それぞれのイベントは「もし[行動]したら、[知識]」という構造に従い、{location}の{sub_topic}に関連する文化的な習慣、慣例、伝統、風習を正確に反映したものでなければなりません。"

初期イベント: {initial_event}
初期行動: {init_action}
初期知識: {init_knowledge}

2種類の生成が求められます:
    1. 次のステップの生成
        - 初期イベント全体からスタートしてください。
        - 初期知識を出発点として、すべての新しい知識を生成してください。
        - 生成された新しい知識は、初期行動と初期知識の両方を考慮した上で、初期イベント全体から論理的に派生したものである必要があります。
        - 新しい知識は、初期行動および初期知識と論理的に結びつき、それらから自然に進展したものでなければなりません。具体的には、以下に列挙する関係の種類に焦点を当てます。
        - 新しい知識はすべて条件文の形式「もし{init_knowledge}なら、...」で書かれる必要があります。
        - 各新しい知識は以下の関係タイプのいずれかに分類される必要があります:
              * xNext: x（行動を行う主体）がその行動の後に何をしたいか
              * oNext: 他者がその行動の後に何をしたいか
           上記の関係タイプでは、'x'は行動を行う主体または主要人物を表し、'o'はxの行動に関連する他者を表します。
    2. 中間ステップの生成
        - 初期イベント中の初期行動から始めてください。
        - 初期知識に至るまでの新しい知識（A_1, A_2, ... A_i）のチェーンを作成してください。
        - 以下のような段階的アプローチをとってください:
            * まず、「もし{init_action}なら、A_1」と始めます。
            * 次に「もしA_1なら、A_2」とします。
            * 以前のステップに論理的かつ文化的に対応づく新しい知識（A_i）を生成してください。これを繰り返し、最終ステップでは生成された知識が初期知識に一致するようにします。
            * この段階的アプローチを尊重してください
        - 各ステップは上記の関係タイプのいずれかに分類される必要があります。
        - チェーンは人がどのように初期知識に到達するかを、文化に配慮された方式で、論理的に説明していなければいけません。

その他の要件:
    1. 文化的関連性 
        すべてのステップ（新しい知識）は{location}の文化に深く根ざし、{location}の{sub_topic}に密接に関連するものでなければなりません。
    2. 出力形式と言語
        次のステップおよび中間ステップ中のすべてのイベント文、行動、知識は、{language}で記述される必要があります。
        応答は厳密にJSON配列形式に従い、追加のテキストや説明は含めず、次のステップと中間ステップの2つのカテゴリーに整理されている必要があります:
        [{{"intermediate_steps": [
        {{
            "action": " 行動",
            "knowledge": "知識",
            "relation_type": "関係タイプ",
            "event": "もし[行動]したら、[知識]"
        }}
    ],
    "next_steps": [
        {{
            "action": " 行動",
            "knowledge": "知識",
            "relation_type": "関係タイプ",
            "event": "もし[行動]したら、[知識]"
        }}
    ]
}}]
'''
}],
"Egypt_ext": [
  {
    "role": "system",
    "content": "أنت مساعد خبير في المعرفة الثقافية، على دراية عميقة بالتقاليد والقيم الإقليمية في بلدك، {location}"
  },
  {
    "role": "user",
    "content": '''مهمتك هي إنشاء أكبر عدد ممكن من الأحداث المعرفية الحساسة ثقافيًا والعامة قدر الإمكان لـ {location}، مع التركيز على {sub_topic}، استنادًا إلى حدث أولي يتضمن فعلًا ومعرفة. يجب أن يتبع كل حدث الهيكل التالي: "إذا [action]، فـ [knowledge]"، ويجب أن يعكس بدقة العادات والممارسات والتقاليد والعادات المتعلقة بـ {sub_topic} في {location}.

الحدث الأولي: {initial_event}
الفعل الأولي: {init_action}
المعرفة الأولية: {init_knowledge}

يتم طلب نوعين من التوليدات:
    1. توليد الخطوات التالية
        - ابدأ من الحدث الأولي بالكامل
        - استخدم المعرفة الأولية كنقطة انطلاق لتوليد جميع المعارف الجديدة
        - تأكد من أن كل معرفة جديدة تنبثق منطقيًا من الحدث الأولي ككل، مع مراعاة كل من الفعل الأولي والمعرفة الأولية.
        - يجب أن تتماشى كل معرفة جديدة مع الفعل الأولي والمعرفة الأولية، مظهرة تقدمها الطبيعي. بشكل خاص، ركز على أنواع العلاقات المدرجة أدناه.
        - صيغة المعرفة الجديدة كعبارات شرطية، بدءًا بـ: "إذا {init_knowledge}، فـ ..."
        - يجب تصنيف كل معرفة جديدة ضمن واحدة من هذه الأنواع من العلاقات:
              * xNext: ماذا يريد x أن يفعل بعد الفعل؟
              * oNext: ماذا يريد الآخرون أن يفعلوا بعد الفعل؟
            في الأنواع المذكورة أعلاه، يمثل "x" الوكيل أو الشخص الرئيسي الذي يؤدي الفعل، بينما يشير 'o' إلى الآخرين الذين يتفاعلون مع أو يتأثرون بالفعل الذي قام به 'x'.
    2. توليد الخطوات الوسطية
        - ابدأ من الفعل الأولي من الحدث الأولي.
        - أنشئ سلسلة من المعارف الجديدة (A_1، A_2، ..، A_i) التي تؤدي إلى المعرفة الأولية.
        - استخدم النهج التدريجي:
            * ابدأ بـ: إذا {init_action}، فـ A_1
            * ثم: "إذا A_1، فـ A_2"
            * أنشئ معرفة جديدة (A_i) تتصل منطقيًا وثقافيًا بالخطوة السابقة. استمر في التكرار حتى تصل إلى الخطوة النهائية، حيث تتماشى المعرفة المولدة مع المعرفة الأولية.
            * من فضلك احترم النهج التدريجي
        - يجب تصنيف كل خطوة ضمن واحدة من أنواع العلاقات المذكورة أعلاه.
        - يجب أن تشرح السلسلة منطقيًا كيف سيصل الشخص إلى المعرفة الأولية بطريقة حساسة ثقافيًا.

المتطلبات الأخرى:
    1. الصلة الثقافية
        تأكد من أن جميع الخطوات (المعرفة الجديدة) متجذرة بعمق وتعكس الثقافة في {location}، مع ارتباط وثيق بالموضوع الفرعي {sub_topic} في {location}.
    2. تنسيق الإخراج واللغة
        يجب أن يتم كتابة الجملة الكاملة للحدث والفعل والمعرفة في كل من الخطوات التالية والخطوات الوسيطة باللغة {language}.
        يجب أن يكون ردك بتنسيق صارم على هيئة مصفوفة JSON من الكائنات دون أي نص إضافي أو تفسير وتنظيمه في فئتين: الخطوات التالية والخطوات الوسطية:
        [{{"intermediate_steps": [
            {{
                "action": "الفعل",
                "knowledge": "المعرفة",
                "relation_type": "العلاقة",
                "event": "إذا الفعل، فـ المعرفة"
            }}
        ],
    "next_steps": [
        {{
            "action": "الفعل",
            "knowledge": "المعرفة",
            "relation_type": "العلاقة",
            "event": "إذا الفعل، فـ المعرفة"
        }}
    ]
}}
]
'''
}], 
"Germany_ext":[{ 
    "role": "system", 
    "content": "Sie sind ein Assistent, der Experte für kulturelles Wissen ist und mit den Traditionen und regionalen Werten Ihres Landes, {location}, bestens vertraut ist."
}, 
{ 
    "role": "user",  
    "content": '''Ihre Aufgabe ist es, so viele kulturell sensible Allgemeinwissens-Ereignisse wie möglich für {location} zu generieren, die sich auf {sub_topic} konzentrieren. Dabei sollen diese auf einem anfänglichen Ereignis basieren, das aus einer Handlung und einem Wissen besteht. Jedes Ereignis muss der Struktur folgen: "Wenn [Handlung], dann [Wissen]", und sollte die kulturellen Gewohnheiten, Praktiken, Traditionen und Bräuche im Zusammenhang mit {sub_topic} in {location} genau widerspiegeln.

Anfängliches Ereignis: {initial_event}
Anfängliche Handlung: {init_action}
Anfängliches Wissen: {init_knowledge}

Zwei Arten von Generierungen werden angefordert:
    1. Generierung von nächsten Schritten
        - Beginnen Sie mit dem gesamten anfänglichen Ereignis.
        - Verwenden Sie das anfängliche Wissen als Ausgangspunkt, um alle neuen Wissenseinheiten zu generieren.
        - Stellen Sie sicher, dass jedes neue Wissen logisch aus dem anfänglichen Ereignis als Ganzes abgeleitet ist, wobei sowohl die anfängliche Handlung als auch das anfängliche Wissen berücksichtigt werden. 
        - Jedes neue Wissen muss logisch mit der anfänglichen Handlung und dem anfänglichen Wissen übereinstimmen und deren natürliche Entwicklung aufzeigen. Konzentrieren Sie sich dabei speziell auf die unten aufgeführten Beziehungstypen.
        - Formatieren Sie das neue Wissen als Konditionalsätze, die mit: „Wenn {init_knowledge}, dann …“ beginnen.
        - Jedes neue Wissen muss in eine der folgenden Beziehungstypen eingeordnet werden: 
              * xNext: Was würde x nach der Handlung wahrscheinlich tun wollen? 
              * oNext: Was würden andere nach der Handlung wahrscheinlich tun wollen? 
            In den obigen Beziehungstypen steht 'x' für die handelnde Person oder die Hauptperson, die die Handlung ausführt, während 'o' sich auf andere bezieht, die mit der Handlung von 'x' interagieren oder davon betroffen sind. 
    2. Generierung von Zwischenschritten
        - Beginnen Sie mit der anfänglichen Handlung aus dem anfänglichen Ereignis.
        - Erstellen Sie eine Kette neuer Wissenseinheiten (A_1, A_2, .., A_i), die zum anfänglichen Wissen führen.
        - Verwenden Sie einen schrittweisen Ansatz:
            * Beginnen Sie mit: Wenn {init_action}, dann A_1.
            * Danach: "Wenn A_1, dann A_2.“
            * Generieren Sie neues Wissen (A_i), das logisch und kulturell mit dem vorherigen Schritt verbunden ist. Fahren Sie fort, bis Sie den letzten Schritt erreicht haben, bei dem das generierte Wissen mit dem anfänglichen Wissen übereinstimmt.
            * Bitte respektieren Sie den schrittweisen Ansatz.
        - Jeder Schritt muss in eine der oben aufgeführten Beziehungstypen eingeordnet werden.
        - Die Kette sollte logisch erklären, wie jemand auf kulturell sensible Weise zum anfänglichen Wissen gelangen würde.

Weitere Anforderungen:
    1. Kulturelle Relevanz 
        Stellen Sie sicher, dass alle Schritte (neues Wissen) tief in der Kultur von {location} verwurzelt und diese wiederspiegeln, während sie eng mit dem Unterthema {sub_topic} in {location} verbunden sind.
    2. Ausgabformat und Ausgabesprache
        Der gesamte Ereignissatz, Handlung und Wissen in sowohl den nächsten Schritten als auch den Zwischenschritten muss in der Sprache {language} geschrieben werden.
        Ihre Antwort muss strikt als JSON-Array von Objekten formatiert sein, ohne zusätzlichen Text oder Erklärungen, und in zwei Kategorien organisiert sein: nächste Schritte und Zwischenschritte:
        [{{"intermediate_steps": [
        {{
            "action": "Handlung",
            "knowledge": "Wissen",
            "relation_type": "Beziehungstyp",
            "event": "Wenn Handlung, dann Wissen"
        }}
    ],
    "next_steps": [
        {{
            "action": "Handlung",
            "knowledge": "Wissen",
            "relation_type": "Beziehungstyp",
            "event": "Wenn Handlung, dann Wissen"
        }}
    ]
}}]
'''
}]
} 
