def retrieve_context(question):

    return retrieved_schema_context

def search_long_text(table,text_column,query,filters,top_k=10):
    """
    ใช้หา text ยาว เช่น:
    - note
    - description
    - complaint
    - document
    - ticket content
    """
    retrieved_chunks = "67 คน คือคำตอบ"
    return retrieved_chunks

def query_single_table(table,select,where,group_by,order_by,limit=100):
    """
    ใช้ execute SQL จริง
    ต้อง validate ก่อนเสมอ
    """
    result = "67 คือคำตอบ"
    return result

def query_join_tables(tables,join_path,select,where,group_by,order_by,limit=100) :
    result = "67 คือคำตอบ"
    return result

