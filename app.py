# === Both Images Uploaded ===
if uploaded_file1 and uploaded_file2:
    st.markdown("---\n### 📁 Both Images Side by Side")
    col_a, col_b = st.columns(2)

    # === Image 1 ===
    with col_a:
        image1, satellite1 = process_image_before_model(uploaded_file1)
        st.subheader("📸 Uploaded Image 1")
        st.image(image1, use_container_width=True)
        st.subheader("🧭 Cropped Satellite 1")
        st.image(satellite1, use_container_width=True)

        with st.spinner("🔧 Generating Roadmap 1..."):
            try:
                tensor1 = transform(satellite1).unsqueeze(0)
                roadmap1 = run_model_on_satellite(tensor1)
                st.subheader("🗺 Predicted Roadmap 1")
                st.image(roadmap1, use_container_width=True)
            except Exception as e:
                st.error(f"❌ Error in Image 1: {e}")

    # === Image 2 ===
    with col_b:
        image2, satellite2 = process_image_before_model(uploaded_file2)
        st.subheader("📸 Uploaded Image 2")
        st.image(image2, use_container_width=True)
        st.subheader("🧭 Cropped Satellite 2")
        st.image(satellite2, use_container_width=True)

        with st.spinner("🔧 Generating Roadmap 2..."):
            try:
                tensor2 = transform(satellite2).unsqueeze(0)
                roadmap2 = run_model_on_satellite(tensor2)
                st.subheader("🗺 Predicted Roadmap 2")
                st.image(roadmap2, use_container_width=True)
            except Exception as e:
                st.error(f"❌ Error in Image 2: {e}")
