package kr.kro.prjectwwtp.persistence;

import java.sql.PreparedStatement;
import java.sql.SQLException;
import java.util.List;

import org.springframework.jdbc.core.BatchPreparedStatementSetter;
import org.springframework.jdbc.core.JdbcTemplate;
import org.springframework.stereotype.Repository;
import org.springframework.transaction.annotation.Transactional;

import kr.kro.prjectwwtp.domain.FlowImputate;
import kr.kro.prjectwwtp.domain.FlowOrigin;
import lombok.RequiredArgsConstructor;

@Repository
@RequiredArgsConstructor
public class FlowInsertRepository {
	private final JdbcTemplate jdbcTemplate;
	
	@Transactional
	public void FlowOriginInsert(List<FlowOrigin> list) {
		String sql = "INSERT INTO flow_origin (time, flowa, flowb, levela, levelb) VALUES (?, ?, ?, ?, ?)";
		
		jdbcTemplate.batchUpdate(sql, new BatchPreparedStatementSetter() {
			@Override
			public void setValues(PreparedStatement ps, int i) throws SQLException {
				ps.setObject(1, list.get(i).getFlowTime());
				ps.setObject(2, list.get(i).getFlowA());
				ps.setObject(3, list.get(i).getFlowB());
				ps.setObject(4, list.get(i).getLevelA());
				ps.setObject(5, list.get(i).getLevelB());
			}
			
			@Override
			public int getBatchSize() {
				return list.size();
			}
		});
	}
	
	@Transactional
	public void FlowImputateInsert(List<FlowImputate> list) {
		String sql = "INSERT INTO flow_imputate (time, flowa, flowb, levela, levelb) VALUES (?, ?, ?, ?, ?)";
		
		jdbcTemplate.batchUpdate(sql, new BatchPreparedStatementSetter() {
			@Override
			public void setValues(PreparedStatement ps, int i) throws SQLException {
				ps.setObject(1, list.get(i).getFlowTime());
				ps.setObject(2, list.get(i).getFlowA());
				ps.setObject(3, list.get(i).getFlowB());
				ps.setObject(4, list.get(i).getLevelA());
				ps.setObject(5, list.get(i).getLevelB());
			}
			
			@Override
			public int getBatchSize() {
				return list.size();
			}
		});
	}

}
