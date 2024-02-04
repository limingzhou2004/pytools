INSERT INTO category_gallery (
  category_id, gallery_id, create_date, create_by_user_id
  ) VALUES ($1, $2, $3, $4)
  ON CONFLICT (category_id, gallery_id)
  DO UPDATE SET
    last_modified_date = EXCLUDED.create_date,
    last_modified_by_user_id = EXCLUDED.create_by_user_id ;