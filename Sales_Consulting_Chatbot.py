import os

from session_control import connect_weaviate
import session_control
from dotenv import load_dotenv

# Load environment variables from .env file
load_dotenv()

os.environ["OPENAI_API_KEY"] = os.getenv("OPENAI_API_KEY")

# Vectorize
from langchain_openai import OpenAIEmbeddings
embeddings = OpenAIEmbeddings(model="text-embedding-3-large")

# Connect to Weaviate Cloud
connect_weaviate()
weaviate_client = session_control.weaviate_client
print(weaviate_client)

from langchain_weaviate.vectorstores import WeaviateVectorStore
docsearch = WeaviateVectorStore(
    embedding=embeddings,
    client=weaviate_client,
    index_name="ChatBot",
    text_key="text"
)

# Pass the premise
premise = """
Giả sử bạn là một chuyên gia về thiết bị điện tử âm nhạc. Bạn đang làm việc tại 769audio, một trong ba nhà phân phối thiết bị âm nhạc hàng đầu tại TP.HCM. Bạn sắp tư vấn cho một khách hàng về phương pháp bán hàng phù hợp nhất để bán hàng cho họ, dựa trên thông tin được cung cấp.
- Bắt buộc trả lời tiếng Việt nếu khách hàng không yêu cầu trả lời bằng ngôn ngữ khác.
- Khi trả lời, vui lòng không thêm 'Dựa trên thông tin được cung cấp' hoặc điều gì đó tương tự.
- Nếu bạn không tìm thấy sản phẩm nào trong dữ liệu cung cấp phù hợp với câu hỏi, vui lòng không sử dụng bất kỳ thông tin bên ngoài nào để điền vào, không được tự tạo sản phẩm và nói rằng không có sản phẩm phù hợp.
- Nếu người dùng đưa ra câu hỏi không liên quan, vui lòng ghi nhận câu hỏi của người dùng, nhưng hướng dẫn nhẹ nhàng.
- Nếu người dùng yêu cầu một bức ảnh, cung cấp cho họ liên kết hình ảnh trong cột "link ảnh". Nếu họ yêu cầu thêm, cung cấp cho họ các liên kết hình ảnh trong cột "tập link ảnh".
- Nếu người dùng hỏi về dung lượng, nếu cột dung lượng không có giá trị nào, hãy tìm dung lượng trong các cột "mô tả" và "giới thiệu".
- Nếu sản phẩm được làm từ Trung Quốc, trước tiên, vui lòng nói rằng nó được nhập khẩu. Nếu người dùng tiếp tục hỏi về nguồn gốc của nó, hãy nói rằng nó đến từ Trung Quốc.
- Nếu họ nói rằng họ muốn mua một cái gì đó, hãy giới thiệu mặt hàng phù hợp cho họ.
- Nếu họ yêu cầu một sản phẩm với giá cả, hãy tìm một sản phẩm với giá gần nhất mà họ cung cấp.
- Nếu bạn phát hiện tên sản phẩm, vui lòng phân tích nó nếu cột tên sản phẩm có sản phẩm đó. Ví dụ, bạn phát hiện "Vang số Karaoke JBL KX180A black", nhưng trong cột tên sản phẩm, có "Vang số JBL KX 180A", điều đó có nghĩa là chúng là cùng một sản phẩm.
- Luôn luôn theo sát sản phẩm trong cuộc trò chuyện đang nhắc đến. Khi khách hàng hỏi về 1 yếu tố nào của sản phẩm mà không đề cập đến sản phẩm, thì bạn phải tự ngầm hiểu là họ đang hỏi về sản phẩm gần nhất trong cuộc trò chuyện đang nhắc tới.
- Khi khách hàng hỏi trang web của 1 sản phẩm nào, nghĩa là họ đang tìm link sản phẩm, bạn phải cung cấp link sản phẩm đúng với sản phẩm trong cuộc trò chuyện đang nhắc tới. Bắt buộc không được tự ý tạo link khác (trang web = link sản phẩm).
- Khi giới thiệu giá cho khách hàng, phải kiểm tra thông tin giảm giá hoặc khuyến mãi trong cột "Giới thiệu" và "Chi tiết" trước khi báo giá cho khách. Và khi báo giá, bạn phải cung cấp giá gốc ban đầu trong cột "giá", rồi mới giới thiệu giảm giá với giá mới trong phần "chi tiết".
- Không được tự tạo link sản phẩm hay bất cứ thông tin nào không có trong tập thông tin được cung cấp. Nếu không có thông tin, cứ trả lời là không có.
- Khi bạn gợi ý sản phẩm cho khách hàng, hãy tìm kiếm các sản phẩm có khuyến mãi/giảm giá trước. Nếu không có thì mới hãy kiếm các sản phẩm còn lại.
- Khi báo giá cho khách hàng, phải lấy thông tin giá từ cột "Giá" hoặc page_content có metadata "source": "Giá". Nếu không có hoặc sản phẩm là Dàn hoặc Bộ dàn, thì bạn mới phải lấy thông tin giá trong chi tiết sản phẩm.
- Khi gợi ý sản phẩm cho khách hàng, nếu tình trạng của sản phẩm là 1, tức là có hàng, thì mới hãy gợi ý, còn không thì không được đề cập tới, trừ khi khách hàng hỏi cụ thể sản phẩm ấy.
- Nếu câu trả lời của bạn đề cập đến một sản phẩm, bạn PHẢI kèm theo đường link sản phẩm trong dữ liệu được cung cấp.
    Ví dụ:
    - Câu hỏi: "Cho tôi biết giá loa JBL 201 Series 4"
    - Trả lời: "Dạ, hiện tại, giá của loa là $$$:
    [<sản phẩm được đề cập>](https://example.com/product/201-series-4)"
- Câu trả lời cần phải được cách dòng 1 cách hợp lý, không thể để các câu đều nằm trên 1 hàng.
- Bạn cũng có thể để các ký hiệu Markdown 1 cách thoải mái và hợp lý, không nhất thiết lúc nào cũng phải là plaintext.

* Đối với việc gợi ý sản phẩm cho khách hàng:

"""
sale_methodology = """
Bạn phải tuân theo phương pháp bán hàng này:
Chúng ta cần thu thập ít nhất 3 tiêu chí của nhu cầu của khách hàng để tư vấn sản phẩm phù hợp nhất cho họ:
1. Thể loại
2. Giá cả
3. Loại phòng
4. Diện tích phòng
5. Mục đích sử dụng / Nghề nghiệp (hoặc công việc của họ)
...
Khi họ đưa ra một tiêu chí (Ví dụ: Tôi muốn mua một sản phẩm giá 500 USD, có nghĩa là một tiêu chí là Giá và là 500 USD),
bạn trả lời họ bằng một câu trả lời liên kết (À! Tôi hiểu! Bạn phải đang tìm kiếm một sản phẩm cho mục đích kinh doanh),
sau đó bạn đưa ra cho họ một câu hỏi dựa trên câu trả lời liên kết mà bạn đã cung cấp (Vậy bạn muốn mua sản phẩm này để làm gì?),
hoặc bạn chỉ cần đưa ra một câu hỏi khác để tìm hiểu tiêu chí khác (Bạn thường nghe thể loại nhạc nào?).
Lặp lại điều này cho đến khi bạn có được nhiều hơn 3 tiêu chí, sau đó bạn đưa ra sản phẩm tốt nhất dựa trên những tiêu chí đó.

Nếu họ có vẻ không quan tâm đến việc tuân theo quy trình, vui lòng chỉ trả lời các câu hỏi của họ.
Nếu họ cung cấp một sản phẩm cụ thể, cung cấp cho họ một số thông tin về nó, sau đó tuân theo phương pháp bán hàng.
"""
objection_handling = """
Nếu khách hàng đưa ra phản đối với sản phẩm của bạn, vui lòng tuân theo quy trình này:
Ghi nhận: Thể hiện sự đồng cảm với sự phản đối của họ (Ví dụ: Tôi hiểu phản đối của bạn, phải khó khăn lắm mới biết giá sản phẩm cao hơn bạn mong đợi)
Khám phá: Hiểu lại các tiêu chí mà họ đã cung cấp (Ví dụ: Nhưng hãy xem xét lại đi. Tôi nghĩ giá trị của sản phẩm này đáp ứng kỳ vọng cao của bạn về giá trị mà bạn mong muốn)
Giới thiệu: Cung cấp cho họ một giải pháp có giá trị mà nếu họ không thực hiện, họ sẽ ân hận, và họ có thể tự thực hiện (Ví dụ: Nếu tôi là bạn, tôi sẽ trả thêm một chút tiền để không phải hối tiếc mỗi đêm vì sản phẩm chất lượng thấp)

Nếu họ tiếp tục phản đối với cùng một lý do, loại bỏ tiêu chí ít quan trọng nhất và giới thiệu lại, hoặc nếu không, nói rằng bạn không thể cung cấp sản phẩm phù hợp.
"""

import pandas as pd
data = pd.read_excel('User Feedback.xlsx')
# feedback_df = pd.DataFrame(data.iloc[1:].values, columns=data.iloc[0])
feedback_df = pd.DataFrame(data)

from langchain.docstore.document import Document

def retrieve_and_filter_chunks(row_numbers, data, excluded_columns=["Giới thiệu", "Chi tiết"]):
    filtered_chunks = []

    for row_number in row_numbers:
        # Check if row number is valid before accessing
        if row_number in data.index:
            row_data = data.loc[row_number]
            for col in data.columns:
                if col not in excluded_columns:
                    filtered_chunks.append(
                        Document(page_content=str(row_data[col]),
                                 metadata={"source": col,
                                           "row": row_number}))
    return filtered_chunks

os.environ["LANGCHAIN_API_KEY"] = "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"
os.environ["LANGCHAIN_TRACING_V2"] = "true"
os.environ["LANGCHAIN_PROJECT"] = "Sales Consulting ChatBot"

from langchain.chains import create_retrieval_chain, create_history_aware_retriever
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain.chains.combine_documents import create_stuff_documents_chain

# Rank documents based on relevance to the query
def rank_documents_by_relevance(query, documents):
    def compute_score(doc):
        name = doc.metadata.get("Tên", "").lower()
        return name.count(query.lower())  # Higher count means more relevant

    return sorted(documents, key=compute_score, reverse=True)

def retrieve_and_combine_documents(query, chat_history, data, retriever):
    initial_docs = retriever.invoke(query)

    # Rank documents by relevance
    ranked_docs = rank_documents_by_relevance(query, initial_docs)

    # Filter documents to prioritize exact matches in metadata
    filtered_docs = []
    for doc in ranked_docs:
        if "Tên" in doc.metadata and query.lower() in doc.metadata["Tên"].lower():
            filtered_docs.append(doc)

    # print(filtered_docs)

    # If no exact matches, fall back to all retrieved docs
    if not filtered_docs:
        filtered_docs = initial_docs

    # Extend with other relevant chunks (if needed)
    row_numbers = {doc.metadata["row"] for doc in filtered_docs}
    additional_docs = retrieve_and_filter_chunks(row_numbers, data)
    filtered_docs.extend(additional_docs)

    # print(filtered_docs)

    return filtered_docs


def initialize_rag(llm, data, retriever):
    def wrapped_retriever(input_data):

        input_query = input_data.content
        # print("chat history:", chat_history)

        contextualized_query = contextualize_query(input_query, chat_history)
        # print("New contextualized query:", contextualized_query)

        return retrieve_and_combine_documents(contextualized_query.content, chat_history, data, retriever)
        # chat_history is still passed, and it's from the from_template

    def contextualize_query(query, history):
        # Use the ChatPromptTemplate to reformulate the query
        # This way seems far-fetched, but let's overlook for better good

        prompt = contextualize_q_prompt.format(input=query, chat_history=history)
        return llm.invoke(prompt)

    # contextualize_q_system_prompt = """Given a chat history and the latest user question \
    # which might reference context in the chat history, formulate a standalone question \
    # which can be understood without the chat history. Do NOT answer the question, \
    # just reformulate it if needed and otherwise return it as is.
    # """

    information_replacement = """
    <information> means you have to fill in the appropriate information based on the context of the conversation.
    Eg: We have this retrieved data: loa pasion 10, giá: 10VND, công suất: 100W.
    Answer form: Vâng, chúng tôi có bán Loa JBL Pasion 10 với <các thông tin cần thiết>
    ==> Actual answer: Vâng, chúng tôi có bán Loa JBL Pasion 10 với giá là 10VND và công suất là 100W.
    """

    feedback_content = """
      Here is the feedback of customers. Please learn from this feedback so that you don't repeat your mistakes.
      Learn the correct format after "as the feedback is" so that you can apply the format for other similar questions.
      <information> means you have to fill in the appropriate information based on the context of the conversation.
      You don't have to use the exact content in Correction value, just fill in the appropriate information, unless it requires correct format.
    """

    contextualize_q_system_prompt = """
    Bạn là 1 nhà tư vấn thiết bị âm thanh, gồm loa, micro, mixer, ampli,...

    Dựa trên lịch sử trò chuyện dưới đây và câu hỏi mới nhất của khách hàng, hãy diễn giải câu hỏi sao cho dễ hiểu và liên quan đến ngữ cảnh đã trao đổi.

    1. Sử dụng thông tin từ lịch sử để thêm chi tiết còn thiếu cho câu hỏi mới (nếu có).
    2. Đảm bảo rằng câu hỏi được diễn giải một cách chính xác và ngắn gọn, nhưng vẫn giữ ngữ cảnh từ lịch sử trò chuyện.
    3. Nếu không có đủ thông tin từ lịch sử, hãy diễn giải câu hỏi mới sao cho dễ hiểu nhất mà không cần ngữ cảnh.

    Ví dụ:
    - Lịch sử: "Cho tôi biết giá loa JBL 201 Seri 4"
    - Câu mới: "Gửi tôi link sản phẩm"
    - Diễn giải: "Bạn có thể cho tôi biết link sản phẩm của loa JBL 201 Seri 4 không?"

    4. Nếu câu trả lời của bạn đề cập đến một sản phẩm, bạn PHẢI kèm theo đường link sản phẩm trong dữ liệu được cung cấp.

    Ví dụ:
    - Câu hỏi: "Cho tôi biết giá loa JBL 201 Seri 4"
    - Trả lời: "Dạ, hiện tại, giá của loa là $$$:
    [<sản phẩm được đề cập>](https://example.com/product/201-series-4)"

    Lịch sử trò chuyện:
    {chat_history}

    Câu hỏi mới: {input}

    Diễn giải:
    """

    # Accumulate corrections based on feedback dataframe
    for index, row in feedback_df.iterrows():
        if pd.notna(row['Correction']):
            feedback_content += f"""
            If a user asks: \"{row['Query']}\", you shouldn't answer like this: \"{row['Response']}\",
            as the feedback is {row['Feedback']}, but you should answer: {row['Correction']}\n\n
            """

    input_premise = """
    - Không được trả lời lại câu yêu cầu, chỉ diễn giải lại nó sao cho phù hợp với những gì được fine-tuned.
    - Bắt buộc sử dụng tiếng Việt nếu khách hàng không yêu cầu ngôn ngữ khác.
    - When customers ask for product suggestions, paraphrase the customer's question to include discount keywords, to prioritize finding products with discounts. If there are no suitable promotional products, then choose the remaining products.    """

    contextualize_q_prompt = ChatPromptTemplate.from_messages(
        [
            ("system",
             contextualize_q_system_prompt + "\n\n"
            #  + contextualized_prompt +  "\n\n"
             + input_premise +  "\n\n"
             + information_replacement
             ), # premise +  "\n\n" + feedback_content + "\n\n" + "\n\n" +  keyword_feedback
            MessagesPlaceholder("chat_history"),
            ("human", "{input}"),
        ]
    )

    # Create a history-aware retriever using the custom wrapped retriever
    history_aware_retriever = contextualize_q_prompt | llm | wrapped_retriever

    qa_prompt = ChatPromptTemplate.from_messages(
            [
                ("system", premise +  "\n\n" + information_replacement + "\n\n" + feedback_content +  "\n\n" + "{context}"),
                MessagesPlaceholder("chat_history"),
                ("human", "{input}"),
            ]
        )

    # Initialize memory and QA system
    question_answer_chain = create_stuff_documents_chain(llm, qa_prompt)

    # Create and return the RAG chain
    return create_retrieval_chain(history_aware_retriever, question_answer_chain)

# Example usage
from langchain_openai import ChatOpenAI

llm = ChatOpenAI(model='gpt-4o-mini', temperature=0)
retriever = docsearch.as_retriever(
    search_type="similarity_score_threshold",
    search_kwargs={
        "score_threshold": 0.4,
        "k": 20
        }
    )

from langsmith import Client

client = Client()

csv_path = "Finished_Data_in_769audio_vn.csv"

data = pd.read_csv(csv_path, encoding="utf-8")
# print(feedback_df.head())
chat_history = []
qa = initialize_rag(llm, data, retriever)

# from langchain_core.messages import HumanMessage, AIMessage
#
# query = "201 seri 4 nhieu vay chi"
# chat_history = [
#     HumanMessage(content="201 seri 4 nhieu vay chi"),
#     # AIMessage(content="Dạ, hiện tại bên chúng tôi có bán Loa Bose 201 seri IV với giá là 4,200,000 VNĐ."),
#     # HumanMessage(content=query)
# ]
# chat_history.extend([HumanMessage(content=query)])
#
# rag = initialize_rag(llm, data, retriever)
#
# from langchain import callbacks
#
# with callbacks.collect_runs() as cb:
#   result = rag.invoke({"input": query, "chat_history": chat_history})
#   run_id = cb.traced_runs[0].id
#
# print(result['answer'])
# print(run_id)
