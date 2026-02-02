package kr.kro.prjectwwtp.persistence;

import java.sql.PreparedStatement;
import java.sql.SQLException;
import java.util.List;

import org.springframework.jdbc.core.BatchPreparedStatementSetter;
import org.springframework.jdbc.core.JdbcTemplate;
import org.springframework.stereotype.Repository;
import org.springframework.transaction.annotation.Transactional;

import kr.kro.prjectwwtp.domain.TmsOrigin;
import lombok.RequiredArgsConstructor;

@Repository
@RequiredArgsConstructor
public class TmsOriginInsertRepository {
	private final JdbcTemplate jdbcTemplate;
	
	@Transactional
	public void TmsOriginInsert(List<TmsOrigin> list) {
		String sql = "INSERT INTO tms_origin (time, toc, ph, ss, flux, tn, tp) VALUES (?, ?, ?, ?, ?, ?, ?)";
		
		jdbcTemplate.batchUpdate(sql, new BatchPreparedStatementSetter() {
			@Override
			public void setValues(PreparedStatement ps, int i) throws SQLException {
				ps.setObject(1, list.get(i).getTmsTime());
				ps.setObject(2, list.get(i).getToc());
				ps.setObject(3, list.get(i).getPh());
				ps.setObject(4, list.get(i).getSs());
				ps.setObject(5, list.get(i).getFlux());
				ps.setObject(6, list.get(i).getTn());
				ps.setObject(7, list.get(i).getTp());
			}
			
			@Override
			public int getBatchSize() {
				return list.size();
			}
		});
	}

}
